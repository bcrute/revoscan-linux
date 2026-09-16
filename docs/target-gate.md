# The target gate: measured against CAD

Status: **first physical agreement measurement this project has produced.**
Every accuracy number before this one was self-consistency (loop closure) or
synthetic (the surface oracle, which says so itself). This one compares a real
scan against real reference geometry.

Run against tolerances declared in
`reference/targets/reconstruction_v1/gates/tolerance-v1.json` **before any
comparison was made**, on capture `20260901-201341` — 190 frames, decimate 2,
190/190 placed, 1.20° held-out loop closure.

## Result

```
ACCURACY  (over observed points)   RMS 1.32 mm   p95 2.51 mm   20/23 surfaces pass
COMPLETENESS                       79% of CAD points observed
```

| surface | CAD pts | observed | RMS mm | p95 mm | |
| --- | --- | --- | --- | --- | --- |
| step_10mm_top | 126 | 100% | **0.59** | 0.88 | PASS |
| pocket_floor | 55 | 100% | **0.68** | 0.98 | PASS |
| step_5mm_top | 110 | 100% | 0.70 | 1.10 | PASS |
| step_10mm_side | 98 | 100% | 0.84 | 1.28 | PASS |
| step_5mm_side | 42 | 100% | 0.92 | 1.65 | PASS |
| pocket_wall | 278 | 100% | 0.96 | 1.99 | PASS |
| step_riser | 285 | 100% | 0.97 | 1.96 | PASS |
| step_2mm_top | 115 | 100% | 1.04 | 1.74 | PASS |
| step_20mm_side | 561 | 100% | 1.05 | 2.05 | PASS |
| step_2mm_side | 49 | 100% | 1.23 | 1.80 | PASS |
| plateau_side | 1406 | 85% | 1.24 | 2.51 | PASS |
| smooth_bump | 560 | 86% | 1.30 | 2.39 | PASS |
| base_top | 2024 | 77% | 1.34 | 2.38 | PASS |
| rail_6mm_side | 117 | 100% | 1.36 | 2.25 | PASS |
| rail_6mm_top | 65 | 100% | 1.38 | 2.10 | PASS |
| plateau_top | 695 | 89% | 1.38 | 2.76 | PASS |
| outer_east | 307 | 86% | 1.38 | 2.57 | PASS |
| ramp_side | 2833 | 83% | 1.40 | 2.62 | PASS |
| rail_3mm_side | 78 | 87% | 1.45 | 2.66 | PASS |
| ramp_top | 981 | 56% | 1.47 | 2.76 | PASS |
| outer_south | 338 | 64% | 1.60 | 2.80 | FAIL |
| step_20mm_top | 162 | 99% | 1.78 | 2.38 | FAIL |
| outer_north | 335 | **19%** | 1.88 | 2.91 | FAIL |
| **rail_1_5mm_side** | 56 | **0%** | — | — | **NOT OBSERVED** |
| **outer_west** | 293 | **0%** | — | — | **NOT OBSERVED** |

## Accuracy and completeness are two questions

A CAD-side nearest-neighbour distance is large for **two unrelated reasons**: a
surface measured badly, and a surface never measured. Reporting one number
conflates them, and the first version of this gate did exactly that — it read
`outer_west` at 10.45 mm RMS and I took it for grazing-incidence error, when
that face has **no scan point within 3 mm of it anywhere**.

So they are reported apart:

- **completeness** — the fraction of a surface's CAD points with any scan point
  inside the 3 mm observation radius;
- **accuracy** — deviation over *only* those points, the only subset where "how
  far off" means anything.

This is the split `docs/coverage-and-subject.md` §3 argues for, and the same one
`mapper` settled on: two verdicts that are never fused.

## What the numbers say

**Accuracy is good and consistent.** 1.32 mm RMS across the whole target is
about 2× the 0.65 mm measured per-point noise floor, and it agrees with an
independent estimate: the 1.20° held-out loop closure contributes ≈1.29 mm at
the target's corner. Two unrelated measurements landing on the same figure is
the strongest evidence here that both are describing something real.

**The fine features survive.** Every one of the four steps is inside 1.23 mm on
both faces, the 12 mm deep pocket reads 0.68/0.96, and the 3 mm and 6 mm rails
are inside 1.45 mm. These are the features the target exists to test.

**The three accuracy failures are marginal and two are coverage artefacts.**
`outer_south` (1.60) and `outer_north` (1.88) sit just past a 1.50 bar, and
`outer_north` is computed from 19% of its points. `step_20mm_top` (1.78) is the
only clean failure on a well-observed surface.

**The real deficiency is coverage of specific faces**, and this is the useful
half. `outer_west` at 0%, `rail_1_5mm_side` at 0% and `outer_north` at 19% name
exactly what the capture missed. Compare with the built-in coverage gate, which
refused the same capture with *"0.471 of the occupied surface"* — a number
diluted three-to-one by turntable and room (see
`docs/coverage-and-subject.md`). Querying from the reference side cannot be
diluted: scenery is never queried.

## `outer_west` at 0% was an artefact of the placement, and two explanations were wrong

The sweep is roughly 1.5 revolutions, so every face passed the camera. Three
explanations were considered; the first two were tested and failed, and the
third turned out to be the answer.

**Registration yaw is ruled out.** A wrong yaw would map CAD-west onto a
different physical face — but the target is deliberately asymmetric, so it would
degrade every surface, and the ramp, pocket and bump all sit at 1.30–1.47 mm.

**A limited sensor band is ruled out, and this one was asserted before it was
checked.** The measurement behind it divided returns by all 400 rows of each
column, which plots *the object's silhouette*, not the sensor's sensitivity: the
target spans 329 rows at column 220 and 74 rows at column 460, so of course the
ratio falls off toward its edges. Dividing instead by the rows the object
actually occupies gives a flat **71–79% fill across columns 80–400**, continuing
to ~520. There is no narrow usable band. The whole object is imaged.

| columns | object height (rows) | fill within silhouette |
| --- | --- | --- |
| 80–119 | 207 | 74% |
| 160–199 | 298 | 79% |
| 200–239 | 329 | 78% |
| 280–319 | 312 | 75% |
| 360–399 | 228 | 71% |
| 400–439 | 149 | 62% |
| 440–479 | 74 | 41% |

That correction matters beyond this one face: an earlier version of this
document and of `tools/target_gate.py` claimed the target "cannot be fully imaged
by this scanner at any distance inside its working range" and used that to
justify not gating completeness. **The claim was false.** Completeness is still
not gated, but for the honest reason: what limits it is unknown.

**It was the placement.** Re-running the same capture through the same gate with
a different seed for the placement search reads `outer_west` at **96%**, not 0%.
Nothing about the scan changed. The per-surface figures move with where the
search happens to land, while the aggregate barely does (1.29 against 1.32 mm
across two independent registrations), because a small residual yaw redistributes
error between opposite faces without changing the total.

An earlier draft of this section spent three paragraphs theorising about
projector shadowing — the projector and camera being separate apertures, the
ramp rising to 56 mm and occluding a side face — and offered the east/west
asymmetry as supporting evidence. **That was explaining an artefact.** The
asymmetry it set out to explain was not a property of the scanner at all, and
the theory was plausible enough to have survived unchallenged had the seed never
been varied. It is recorded here rather than deleted because the failure mode is
the point: a coherent physical story fitted to one registration's output.

The consequence for this gate is concrete: **a per-surface verdict is not
trustworthy from a single placement.** Either report per-surface numbers with a
spread across seeds, or report only the aggregate.

## What the gate's number is made of, and why it is not the declared one

Five measurements, in the order they were forced. Each was provoked by the one
before disagreeing with what it was supposed to show.

**The fused surface is thin.** Local plane RMS in a 2 mm neighbourhood of the
fused cloud is **0.669 mm**, p90 0.787 - essentially the 0.65 mm per-point noise
floor, with no heavy tail. There is no ghosting and no doubled surface. Whatever
separates the reconstruction from the CAD is systematic, not smear.

**More laps make it worse, not better.** The plan was that loop closure limits
accuracy and revisit constraints improve closure, so a longer sweep should help.
A 450-frame, 3.5-lap capture reconstructs *worse* than a 190-frame, 1.5-lap one:
1.40 mm against 1.22 mm, 12/23 surfaces against 20/24, completeness 70% against
80%, and the support-plane margin collapses from 1.05x to 1.01x. Sequential
drift accumulates faster than the graph's revisit constraints remove it. This
route is closed.

**Freeing the scale buys nothing.** Registering with a similarity rather than a
rigid transform changes RMS by 0.001 mm (1.304 -> 1.303) while pulling the scale
to 0.986. A -1.4% shrink that improves no residual is ICP's shrink-to-fit
degeneracy, not a depth-scale error. The tolerance file's *conclusion* - that
nothing here tests the depth scale - is correct. Its stated *reason* is not:
"rigid registration absorbs any global scale error" is false, because a rigid
transform has no scale parameter to absorb one with. The right reason is that
the residual is insensitive to scale at this magnitude.

**The gate's metric is not the declared metric, and it is biased.** The file
declares "RMS deviation of registered scan points from their own CAD *surface*"
- a point-to-surface distance. `target_gate.py` measures CAD-point to nearest
scan-*point* Euclidean distance. Against a synthetic scan carrying a *known*
perpendicular error, the two behave very differently:

| true perpendicular error | euclidean (gate) | scan-side perpendicular |
| --- | --- | --- |
| 0.00 mm | 0.34 / 0.64 | 0.143 / 0.144 |
| 0.30 mm | 0.41 / 0.70 | 0.325 / 0.325 |
| 0.60 mm | 0.49 / 0.79 | 0.590 / 0.592 |
| 1.00 mm | 0.57 / 0.90 | 0.938 / 0.928 |

(two figures per cell: scan sampled at 4x and 1x the CAD density)

The declared quantity is recovered by the scan-side form and overstated by the
gate's. Worse, the gate's number **nearly doubles when the scan is made denser
at zero true error** (0.338 -> 0.639): part of what it reports is point density,
which is a property of the sampling and not of the scanner.

**But the declared metric cannot be measured on this scene.** Scan-side means
deciding which scan points are on the target, and the numbers say that decision
is not available here. The scan-side RMS is a fixed fraction of whatever
observation radius is chosen - 1.56 mm at 3 mm, 1.05 at 2 mm, 0.77 at 1.5 mm,
0.48 at 1.0 mm, a ratio of 0.48-0.53 throughout. An orientation filter, which
fixed exactly this problem for the orthogonality gate, does **not** fix it here.

Scenery was the obvious explanation and it is **wrong**. If the contaminant were
the turntable, surfaces standing clear of it would behave differently, and they
do not: `ramp_top`, 17.2 mm above the support plane, tracks the cutoff at
x0.55/x0.54/x0.53 while `base_top`, 9.8 mm below it, tracks it at
x0.50/x0.51/x0.50. Every surface tested gives the same ratio at every cutoff
regardless of height. A uniform result across the whole object is not
contamination from one part of the scene.

Two CAD-side substitutes were tried and both are biased, in opposite directions.
Projecting the nearest scan point onto the CAD normal *understates* by about
half (0.32 where the truth is 1.00), because choosing the nearest point
preferentially chooses one whose perpendicular offset is small. Averaging signed
offsets over a neighbourhood understates worse (0.19 where truth is 1.00),
because averaging zero-mean noise measures the surface's *bias* rather than its
spread.

So the position is: **the gate's current number overstates the declared quantity
by an unknown amount, and no unbiased substitute is available without segmenting
the target from the turntable.** Segmentation has failed twice here (see Method
below). This is the blocker, and it is a measurement problem rather than a
scanner problem.

## CORRECTION: the fusion produces a volume, and registration is a dominant term

**The section below is wrong and is kept because the way it went wrong is the
useful part.** It concluded that per-frame geometry, not registration, limits
agreement with the CAD. That rested on a single frame's residual against the CAD
being as large as the whole fusion's - measured through a frame-to-fusion
registration whose own median was 1.6-2.3 mm, which is the same size as the
quantity it was measuring. The caveat was written down at the time and then
relied on anyway.

The direct test is the SHAPE of a local neighbourhood, not the size of one
extent. Median principal extents of a 2 mm neighbourhood, against two
references measured the same way:

| cloud | extents (mm) | λ3/λ2 | |
| --- | --- | --- | --- |
| reference: thin surface, 0.15 mm | 1.019 / 0.956 / 0.148 | **0.155** | what a surface reads |
| reference: isotropic volume | 0.738 / 0.731 / 0.645 | **0.883** | what a blob reads |
| one prepared frame (r = 6 mm) | 3.047 / 2.370 / 0.483 | **0.204** | a surface |
| fused, `20260902-224804` | 0.972 / 0.866 / 0.679 | **0.784** | nearly a volume |
| fused, `20260901-201341` | 0.982 / 0.851 / 0.683 | **0.803** | nearly a volume |

A single depth frame *cannot* be volumetric - one depth per pixel makes it a
surface by construction, and it measures 0.204. **The fusion is what turns it
into a volume**, so the frames are being placed inconsistently and registration
is a dominant error term.

This also retires an earlier claim in this document: the fused cloud was
reported "locally 0.669 mm thin" because that number sits near the 0.65 mm
per-point noise floor. The comparison was against the wrong reference. Points
filling a 2 mm ball uniformly give about 0.89 mm by that statistic and a genuine
surface gives 0.1-0.2 mm, so 0.669 was always much closer to a blob. One extent
cannot tell a thin surface from a thick one; only the ratio can.

Everything that follows from "registration is not the limiter" is affected,
including the reading that the 450-frame sweep and the coarse-to-fine schedule
failed because they addressed a non-dominant term. What they show is narrower:
those particular interventions did not thin the cloud.

## The ceiling is per-frame geometry, not registration

This is the finding that reorganises the rest. Every route pursued in this
session assumed the error was accumulated across frames - better closure, more
laps, tighter refinement. A single frame carries no pose graph at all, so
comparing one against the CAD separates the two.

Registered through the fusion (the fused cloud places on the CAD at 0.96 mm, and
a single frame registers to the fused cloud because it is the same scene at full
coverage), so the frame's own points reach CAD coordinates having undergone
exactly ONE rigid motion:

| | residual RMS vs CAD | by image column (axis at column 42) |
| --- | --- | --- |
| frame 47 | 1.686 mm | 1.26 / 1.40 / 1.57 |
| frame 95 | 1.823 mm | 1.66 / 1.59 / 1.60 / 1.76 |
| frame 142 | 1.921 mm | - / 1.71 / 1.82 / 2.63 |
| **fused, 190 frames** | **1.767 mm** | |

(all scan-to-CAD Euclidean over points within 3 mm, so the rows are comparable)

**A single frame is as far from the CAD as the entire fusion.** The three frames
bracket the fused figure rather than sitting below it. Fusion adds no error and
removes none, which means registration quality is not what limits agreement -
and that retrospectively explains two results that were otherwise puzzling: why
a 450-frame sweep reconstructs *worse* than a 190-frame one, and why
coarse-to-fine refinement bought 0.08 mm. Both were working on a term that is
not the dominant one.

**The residual grows with distance from the optical axis.** `cx = 42.44` puts
the axis near the frame's left edge, so column is the axis-distance proxy, and
all three frames rise left to right - most sharply on frame 142, 1.71 to 2.63 mm.
That is the signature of an unmodelled radial term.

What this does NOT establish, and the caveats are load-bearing:

- **The frame-to-fusion registration is loose** - median 1.6-2.3 mm at matched
  fraction 0.67-0.77 - so the absolute residuals carry registration error of the
  same order as the quantity being measured. The comparison between single frame
  and fusion is sound because both are measured the same way; the absolute
  values are not a calibration result.
- **Column is confounded with geometry.** Points at high column are also further
  from the object's centre and at more grazing incidence, and this construction
  cannot separate a radial distortion term from an incidence-angle effect.
- **`fx` and `fy` remain untested.** `docs/principal-point.md` settles `cx` and
  `cy` and says explicitly that it is blind to focal length, because a focal
  error scales the reconstruction without rotating any plane normal.

The actionable consequence is narrow and specific: **lens distortion is
completely unmodelled and has never been read from the device.**
`/data/camparam/Distort.bin` has not been fetched, because reading it detaches
interface 0 from `uvcvideo` and that is a separate authorisation from streaming.
It is now the highest-value unknown in the metric chain, and no amount of
tracking, cadence or pose-graph work will move the gate while it is open.

## Repeatability, and what the bar has to clear

Two captures were taken three minutes apart with identical settings - 190 frames
at decimate 2, same object, same lighting - specifically to remove the confound
in an earlier comparison that used captures of different lengths.

They disagree with each other more than the gate's tolerance would suggest is
comfortable:

| | RMS vs CAD | surfaces inside 1.5 mm | completeness |
| --- | --- | --- | --- |
| A (`20260902-224455`) | 1.46 mm | 16/25 | 65% |
| B (`20260902-224804`) | 1.11 mm | 21/25 | 79% |

B is the best result this project has produced and A is among the worst, three
minutes apart. Session-to-session variation is larger than any improvement
pursued in this session.

Measured properly - at each CAD point, the difference between the two captures'
signed surface offsets, so that the placement error they share cancels -
**scan-to-scan disagreement is RMS 1.024 mm, median 0.592 mm**. That is coherent
with each capture sitting 0.6-0.8 mm from CAD by the same construction, since
sqrt(0.8^2 + 0.6^2) = 1.0.

Repeatability at ~1.0 mm sits *below* the 1.5 mm bar, so the gate is not
arithmetically impossible. But a single capture's verdict is not reproducible at
the per-surface level, which the placement sweep says independently:

| seed | 5 | 11 | 23 | 42 | 97 |
| --- | --- | --- | --- | --- | --- |
| RMS mm | 1.22 | 1.24 | 1.22 | 1.26 | 1.26 |
| surfaces inside 1.5 mm | 20/24 | 21/25 | 22/25 | 17/24 | 19/25 |

The aggregate is stable to +/-0.02 mm. The per-surface count swings by five
surfaces on the same capture and the same code.

## The depth scale is linear, and one feature dimension is very good

Two more candidates closed, one firmly and one only partly.

**Depth linearity.** A scale that varies with `z` is not a global scale error -
rigid registration cannot absorb it and neither can pose work, because it
deforms every frame identically. That made it a strong candidate for the
per-frame ceiling above, and the repository records a −0.49% to −2.26% depth
scale spread *across distances* as an open question. Swept registration-free by
the plane-angle test, under `z' = z·(1 + a·(z − 200)/100)`:

| a (%/100 mm) | −1.0 | −0.5 | **0.0** | +0.5 | +1.0 |
| --- | --- | --- | --- | --- | --- |
| median \|angle − 90°\| | 0.677° | 0.323° | **0.279°** | 0.586° | 0.959° |

One basin at a = 0. **The depth scale is linear** to the resolution of this test,
roughly 0.5% per 100 mm. (The plane-fit residual falls monotonically toward
negative `a` rather than forming a basin, so it does not discriminate here - the
same shrink-everything behaviour the `fx` sweep shows. The angle metric is the
one with a basin, and it is the one the `cy` control validated.)

**Feature dimensions.** A step height is the separation of two parallel planes
*both taken from the same scan*, so a rigid registration error is common to both
and cancels exactly. That measures printed geometry and scanner scale together
with the placement search removed - and it is `tolerance-v1.json`'s third gate,
`feature_survival`, declared "reported, not gated" and never computed until now.

On `20260902-224804`, plateau-to-base against the CAD's own separation:

| scan-side selection | scan points | CAD | scan | error |
| --- | --- | --- | --- | --- |
| 1.5 mm | 1070 | 15.957 | 15.981 | **+0.024 mm** |
| 2.5 mm | 2195 | 15.957 | 15.999 | **+0.042 mm** |

A 16 mm feature reproduced to within 0.04 mm, and stable under a large change to
its own selection rule.

**This is one feature on one capture, and it is not corroborated.** The same
measurement on three other captures - `20260901-201341`, `20260902-224455` and
the 450-frame sweep - reports `plateau_top` not observed densely enough to fit,
because their placements are looser (1.14, 1.85 and 1.60 mm against 0.96 mm) and
fewer scan points fall near the face. Every other feature pair fails the
200-point bar too: the step tops are small, and a 25-point plane read 0.199 mm
against a 2.553 mm feature, which is not a measurement of that step.

Two earlier versions of this measurement were wrong and are worth recording,
because both produced confident numbers:

- The CAD reference planes were selected **by radius** rather than by each
  face's own point mask, so widening the scan-side radius also pulled
  neighbouring faces into the *reference* plane. The quantity being compared
  against moved with a tuning parameter - CAD plateau-to-base read 15.823 mm at
  one radius and 15.649 mm at another - and the reported error swung from
  +0.158 to +0.399 mm.
- `ramp_top` was paired against `base_top` as though it were a step. It is a
  **sloped** face, so a centroid separation along the base normal is not a
  height at all; on 75 points it read −5.447 mm against a nominal 27.323 mm.

What the surviving measurement supports is narrow: at this scale the print and
the scanner's dimensional accuracy are both good, so **the ~1.1 mm the gate
reports is not dimensional error**. What it cannot support is that surfaces sit
where the CAD says, because a bias shared by both faces of a pair cancels in
their difference - which is exactly the quantity this construction discards in
order to remove the registration.

## Why the CAD-side direction is right, established by four failures

The scan-side measurement is heavy-tailed and its RMS tracks whatever cutoff is
chosen. Four explanations were tested and all four failed, and the pattern of
failure is what finally settles the direction the gate should query from.

**Not scenery by height.** `ramp_top`, 17.2 mm clear of the support plane,
tracks the cutoff at x0.55/x0.54/x0.53; `base_top`, 9.8 mm below it, at
x0.50/x0.51/x0.50. Every surface behaves alike regardless of height.

**Not an outlier halo.** Points with fewer than 5 neighbours within 1 mm read
1.542 mm; points with 20 or more read **1.588** - the dense core is no closer to
the CAD than the strays.

**Not the estimator.** On a synthetic scan with a *known* perpendicular error,
a well-measured surface gives a CONSTANT RMS across cutoffs, not a constant
ratio: true 0.30 reads 0.32/0.32/0.32/0.31 at 3.0/2.5/2.0/1.5 mm. Only as the
true error approaches the cutoff does RMS begin to track it. The real captures
track harder than synthetic σ = 1.5 mm, so the spread is real and heavy-tailed.

**Not edge mis-assignment**, which was the best remaining candidate, since this
target is deliberately covered in steps and rails and a scan point near an edge
pairs with a CAD point on the adjacent perpendicular face. Edge-ness computed on
the CAD alone - a point whose 2.5 mm neighbourhood swings more than 20° - covers
45% of the model, and excluding it changes nothing:

| edge exclusion | cut 3.0 | cut 2.0 | cut 1.5 | n |
| --- | --- | --- | --- | --- |
| none | 1.55 | 1.05 | 0.77 | 61926 |
| swing < 20° | 1.60 | 1.09 | 0.78 | 36962 |
| swing < 5° | 1.60 | 1.09 | 0.78 | 35508 |

**What the four failures jointly establish is the asymmetry.** Querying from the
CAD reads 1.11 mm; querying from the scan reads 1.55 mm and worse. Every CAD
point has a scan point near it, and many scan points are not near the CAD
surface. That is extra scan content - the scene the target stands in - and it is
precisely the thing a CAD-side query cannot be inflated by and a scan-side query
cannot be separated from without segmentation.

So the gate's **direction** is correct and is not the defect. The defect is
narrower than the earlier sections suggested: within the CAD-side direction, the
tool measures a Euclidean point-to-point distance where the tolerance file
declares point-to-surface, which overstates by a lateral term of median 0.6 mm
and varies with scan density. Correcting only that would put the aggregate near
sqrt(1.11² − 0.6²) ≈ 0.93 mm, and the honest way to do it is an unbiased
CAD-side perpendicular estimator, which this session did not find: projecting the
nearest scan point understates by half through selection bias, and averaging a
neighbourhood understates worse by cancelling the noise it is meant to measure.

## Five estimators, and why the declared metric is not measurable on this target

`tolerance-v1.json` declares "RMS deviation of registered scan points from their
own CAD **surface**" - a point-to-surface distance. The tool measures
point-to-nearest-point instead. Five constructions were tried to measure the
declared quantity, each validated against a synthetic scan whose perpendicular
error is *chosen* so the right answer is known. All five fail, and the pattern
is what matters.

| estimator | behaviour at known error | why it fails |
| --- | --- | --- |
| Euclidean nearest point (current) | 0.34 at true 0.00 | overstates; adds lateral sampling, and nearly doubles at zero error when the scan is made denser |
| nearest point projected on normal | 0.32 at true 1.00 | understates by half - choosing the nearest point preferentially chooses small perpendicular offset |
| neighbourhood mean of signed offsets | 0.19 at true 1.00 | understates worse - averaging zero-mean noise measures bias, not spread |
| scan-side, plane through nearest CAD point | recovers truth on synthetic | unbiased, but on a real scan cannot be separated from the scene the target stands in |
| normal-aligned cylinder, lateral selection only | 2.02 at true 0.00 (h=5) | the cylinder punches through the object into other surfaces |
| cylinder + orientation filter | 0.29/0.30/0.47/0.62 at true 0.00/0.30/0.60/1.00 | flat in h, and correct at 0.30 - but truncates larger errors while a wider cylinder returns a floor at zero |

The fourth is unbiased and the sixth is close, and both are defeated by the same
property: **this target is deliberately feature-dense.** Its whole purpose is
steps of 2, 5, 10 and 20 mm and rails of 1.5, 3 and 6 mm, so no local
neighbourhood of a CAD point contains only that point's surface. Every attempt
to exclude the intruders either truncates the distribution being measured
(understating) or leaves a floor of them behind (overstating), and no cylinder
half-height does neither: at 1.5-2.0 mm it truncates, at 3.0 mm it leaves 0.289
where the truth is zero.

**So the declared metric is not measurable on this target by any construction
found here**, and the tool keeps the Euclidean CAD-side distance, which the four
failures above establish is at least the right *direction*: extra scan content
cannot inflate it. Its known defect stands - it overstates by a median 0.6 mm
lateral term and varies with scan density - and it is stated in the tool's own
output rather than silently corrected.

The consequence for the gate is worth being plain about. The 1.5 mm bar was set
against a metric that overstates by an unquantified amount, so **the gate as
written is stricter than it reads**, and the honest fix is a target with a large
uninterrupted flat face to measure against - not a change to the bar, and not an
estimator this session could defend.

## The route the correction opens, and how far it got

If the fusion turns surfaces into a volume, the fix is a representation that
averages frames instead of stacking them. A truncated signed-distance field is
exactly that - each frame votes on the distance to the surface at each voxel and
the votes are averaged, so pose disagreement blurs the zero crossing rather than
thickening a cloud - and this repository already has one, with the same
resolution chooser and constants the export path uses.

It runs. On `20260902-224804`: 190 poses, a 253 x 205 x 277 field at 0.96 mm
voxels, all 190 raw frames integrated, and a surface of 102,063 vertices and
167,289 faces. The extracted surface **is measurably thinner** than the stacked
cloud by the same neighbourhood-shape statistic:

| cloud | λ3/λ2 | |
| --- | --- | --- |
| stacked point cloud | 0.783 | nearly a volume |
| TSDF extracted surface | **0.525** | between, and about half of what remains is the 0.96 mm voxel grid |

**But it does not yet measure against the CAD, and the reason is not settled.**
The gate refused to place the CAD onto it at all - 23.2% of CAD points reached
the surface, under the 25% the tool requires - which is the registration guard
working, since a closed isosurface carries a front and a back face on every thin
structure and the placement search is seeded from a support plane and a density
peak. Holding the placement fixed from the stacked cloud instead, so only the
representation changes, the surface is in the right frame (extents agree to a
few mm) and 45,494 of its vertices sit within 60 mm of the target centre - yet
CAD-to-surface distance has a **median of 18.80 mm**, with p10 5.38 and a
minimum of 0.07.

So the surface is present near the object and mostly not *on* it. That is a
structural fault in how the field is being driven, not a quality difference, and
the candidates are specific: `extract_surface` documents itself as returning the
**connected** zero-level surface, so a component other than the target may be
what came back; and a turntable sweep is a case where the subject rotates while
the room does not, so free-space carving sees a background that is inconsistent
between frames in the model frame.

Recorded as unfinished rather than as a result. What it does establish is that
the machinery runs end to end on a real capture at a usable resolution, and that
the representation really is thinner - which is the premise the correction above
predicts.

## Method

Isolation was the wrong problem and two attempts at it failed before that was
clear:

1. **Plane-based** — fit the turntable, keep what stands on it. Gave 211,507
   points for a 92 × 82 × 56 mm object, an ambiguous coarse alignment and ICP
   rmse 3.9. The numbers were meaningless and were discarded rather than
   reported.
2. **Connected components** at 4 mm cells — returned **one** component of
   404,379 points spanning 385 mm. Target, turntable and room are one touching
   body and no cell size separates them. `support_plane` does not help: it has
   never fired on any real capture (see `docs/support-plane.md`).

The construction that works needs no segmentation at all:

- Place the CAD coarsely — the dominant plane fixes *up*, a 2-D density peak
  fixes the centre, and yaw is searched in 3° steps.
- Then alternate: select scan points within 5 mm of the placed model, refine the
  model onto those with point-to-plane ICP, repeat. Convergence on this capture
  ran 4.21 → 2.94 → 2.41 → 2.15 → 1.90 → 1.85 mm median with matched fraction
  rising 0.68 → 0.94.
- Query from the CAD side. Extra scan content is never queried, so it can
  neither inflate nor deflate the result.

## Limitations

- **Not an accuracy claim.** The print record is
  `ready_for_formal_physical_gate: false` — caliper resolution, printer
  parameters and repeat readings are unrecorded, so the ground truth's own
  uncertainty is unmeasured. This is a development metric against
  CAD-plus-predeclared-shrinkage.
- **Blind to scale.** Rigid registration absorbs any global scale error, so
  nothing here tests the depth scale. The −0.49% to −2.26% spread across
  distances is untouched and still open.
- **The 3 mm observation radius is a choice, not a measurement.** It separates
  "seen" from "not seen" and was not declared in the tolerance file, because the
  need for the split only became clear once the gate ran. It should be declared
  before the next run.
- **One capture, one print, one session.** Nothing here says the result repeats.


## 2026-09-03: the smear, the room, and the first coherent sweep

Everything above from "What the gate's number is made of" onward was measured on
registrations that were not coherent. This section records how that was found, what
it overturns, and what the repository now does about it.

**The ground truth was in the images.** A turntable sweep repeats its depth images
every revolution. Matching each sampled frame against every later one by median
absolute depth difference finds the period - 126 frames on `20260902-224804`, 127 on
`20260903-142612` - and frames one period apart agree to 0.15 mm, better than adjacent
frames do (0.3-0.4 mm). Two identical images must have the same pose, so a registration
can be judged with no CAD at all: carry frame k+P's points through pose k+P and back
through pose k, and read frame k's own depth there.

| capture | registration | step per frame | poses one period apart | frame k+P landing in frame k within 1.5 mm |
| --- | --- | --- | --- | --- |
| `20260902-224804` (previous best) | as committed | 2.67 deg | up to 24 cm apart | 17-99%, two of four pairs under 25% |
| `20260902-224804` | 130-275 mm clip | 2.89 deg | up to 11 cm apart | 47-97% |
| `20260903-142612` | as committed | 2.67 deg | up to 62 deg apart | 114 of 190 frames off the target entirely |
| `20260903-142612` | 130-275 mm clip | 2.85 deg | 1.1 deg, 6 mm (p90) | 90-99% |

The true step is 360/126.5 = 2.85 degrees. The committed pipeline recovered 2.67, 94% of
a turn per period, and loop closure then bent the trajectory to make the revisits it was
given agree, leaving everything between them misplaced. Held-out closure read 1.3
degrees on the previous best capture throughout.

**Why no score saw it.** Closure is scored on the revisit pairs the optimiser was handed.
The gate queries from each CAD point to the NEAREST scan point, and a union of misplaced
copies is dense everywhere: the MIRRORED CAD fits the committed `20260902-224804` cloud as
well as the real one (77-80% observed, 1.13 mm, 22/25 surfaces, against 82-83%, 1.14 mm,
22-23/25). A chiral object fitting its own mirror image means the cloud has no shape at
the millimetre level. Per-frame "on target" shares are fooled the same way, because
`base_top` is nearly symmetric under yaw. Overlaying the placed CAD on single frames
shows it 5-30 mm off per face: it fits the union, not any frame.

**The cause is the room.** The bottle and the cylinder standing behind the turntable at
250-300 mm are inside the 60-300 mm clip in every frame, and their decoding phantoms at
60 and 100 mm come and go with the revolution - the "near-field obstruction" of
2026-08-31 was never the optics. Static in the room while the subject turns, they are a
second rigid body, and point-to-plane ICP fits a compromise. Re-registering the SAME
frames with the preparation clipped to 130-275 mm gives, on `20260903-142612`, a camera
orbit that is a circle of radius 215 mm with 0.28 mm residual and poses one period apart
within 1.1 degrees.

**What the coherent sweep measures.** The support-plane seed fails on a coherent cloud -
the desk becomes as clean a plane as the turntable, the margin reads 1.0x, and the
density peak lands on the turntable disc - so the CAD is placed from the orbit's axis
instead. The gate's own metric on `20260903-142612` then reads:

```
ORTHOGONALITY  worst 0.95 deg against a 2 deg gate, 4/4 faces resolvable  -> PASS
ACCURACY  RMS 0.80 mm  p95 1.83 mm  22/24 surfaces inside 1.5 mm
COMPLETENESS  95% of CAD points observed
```

`plateau_top` 0.16, `outer_west` 0.20, `ramp_top` 0.36, the 1.5 mm rail 0.39. The
failures are `step_10mm_side` (1.62) and `pocket_wall` (1.65, 74% seen), and
`pocket_floor` is unobserved (11%) from a viewpoint 20 degrees above the turntable. The
orthogonality gate, declared first in `tolerance-v1.json` and never evaluable before
because every outer face came back as a sliver, now fits all four faces as planes with
0.19-0.39 mm RMS - and `base_top` at 0.25 mm over 36,822 points, which is what this
sensor's per-point noise on a flat face actually is. The TSDF fusion of the same frames
at 0.5 mm voxels extracts a surface at 0.93 mm RMS with 96% coverage and reads 0.10 at
the frames' own points: the fusion was never the problem, its input was.

**What this overturns.** The "fusion produces a volume" correction rested on a statistic
that cannot tell a plane with 0.65 mm noise (it reads 0.76) from a volume; registration
WAS the dominant term, for the reason above and not that one. The 1.1-1.4 mm
"within-frame" scatter on `base_top` was a tilted copy of the plate crossing the 3 mm
selection band; on the coherent registration single frames scatter 0.58 mm on
`base_top` at 60-67 degrees incidence and 0.23-0.30 mm on `plateau_top`. The 450-frame
comparison, the coarse-to-fine finding, the per-frame ceiling and the TSDF section were
all measured on smears. The single-frame intrinsics work (cx, cy, fx, depth linearity)
does not depend on registration and stands.

**With the background removed (2026-09-04).** Matte grey-black cloth about 30 cm
behind the subject took the alias points to zero (see `docs/phantom-aliases.md`),
and on capture `20260904-020845` the DEFAULT 60-300 mm clip registered coherently
and read the best result so far:

```
ORTHOGONALITY  worst 1.45 deg against a 2 deg gate, 4/4 faces resolvable  -> PASS
ACCURACY  RMS 0.63 mm  p95 1.24 mm  23/24 surfaces inside 1.5 mm
COMPLETENESS  95% of CAD points observed
```

Only `pocket_wall` (1.64 mm) fails and `pocket_floor` (16%) is unobserved, both
the deep pocket seen from 20 degrees of elevation. The same capture clipped to
130-275, 130-330 or 60-275 registered INCOHERENTLY (2-7% of revisit points
within 1.5 mm at the loop's start) while every one of those runs reported a
held-out closure of 0.5-0.8 degrees. So the band is not a cure and the
registration is fragile in a way the closure score cannot see; the coherence
check is the guard, and the registration itself is the next thing to fix.

**What the repository now does.** Preparation takes a `DepthBand` (`--depth-band NEAR
FAR`, or `depth_band` under `[registration]` in the config file) and records it on every
cloud; `scanner3d.coherence` reads the image period and the revisit agreement,
`scanner3d reconstruct` reports the verdict, and `tools/target_gate.py` refuses an
incoherent sweep outright and seeds its placement from the camera orbit. The desktop
application carries the band as the **keep depths** controls in its technical bar - live
preview, tracking and Reconstruct all clip to it, the export fuses over the same
interval, and **Remember** writes it to the config file (`scanner3d config
--set-depth-band` does the same from a shell). What it does not yet do: crop to the
turntable region without being told the band, or weight the fusion by incidence.

## 2026-09-04, evening: a different object wobbles at every band

The operator's first reconstruction from the desktop application was of a white
model on the turntable - one large inclined face, a row of fine columns, small
discs on the plate - captured at the adaptive cadence (146 frames, image period
75, about 5 degrees per frame). It registered incoherently at the default band
(4% of points within 1.5 mm at frames one period apart, 13 mm off) and no better
at 130-275 (2%, 15.5 mm) or 100-260 (6%, 8 mm). The chain's total rotation is
right (363 degrees per period), so this is not the under-rotation of 2026-09-03;
the camera centres wobble off their circle instead (radius 211-224 mm between
the 10th and 90th percentiles, orbit residual 3.1 mm median and 7.2 mm p90, 9 mm
out of plane, against 0.28 mm on the coherent sweep), and poses one period
apart disagree by 9-14 mm through most of the turn and by up to 60 mm over the
last ten frames. Sixty revisits were verified and thirty used; the closures do
not hold the chain on the circle. The depth band fixed the room-content failure
and does nothing here. The registration needs the turntable's own structure -
one fixed axis, one radius, and the period pairs as constraints - which
`scanner3d.coherence` already measures.

**Resolved the same evening:** the fix is not a better chain but no chain. A
turntable sweep is one axis and one rate, and `scanner3d.turntable` fits those
five unknowns directly - the axis and centre from alignments between frames
60 degrees apart, the rate from frames one revolution apart found by time.
Through this tool's own path (`tools/target_gate.py CAPTURE`, which now tries
the turntable model first) the CAD capture above scores 0.50 mm RMS, p95
0.89, 23/24 surfaces, 95% observed, orthogonality PASS worst 0.75 degrees,
base plate 0.11 mm; the white model that smeared registers to 0.10 mm between
neighbours and 0.11 mm one turn apart. What it took to get there - the axis
to two tenths of a degree and the centre to half a millimetre, and why chains,
loop closure and the disc's normal all miss that - is in
`docs/turntable-registration.md`.

## 2026-09-11: the metric correction, on the CAD target

The unit's measured metric correction ([metric-validation.md](metric-validation.md):
`fy` × 0.9960, depth scale × 1.0015, from the cue ball) is now applied by the
window, by `scanner3d reconstruct` and by this tool. Run on the CAD capture
`20260904-020845` (190 frames, turntable model, default 60-300 band) both ways,
`tools/target_gate.py CAPTURE --workers 4` against `--no-metric-correction`:

| | corrected | factory record |
| --- | ---: | ---: |
| accuracy RMS over observed CAD points | 0.513 mm | 0.502 mm |
| p95 | 0.902 mm | 0.896 mm |
| surfaces passing the 1.5 mm gate | 23/24 | 23/24 |
| CAD points observed | 95.2% | 95.0% |
| orthogonality, worst outer face to `base_top` | 0.70 deg | 0.75 deg |
| turntable model | 126.67 frames/turn, axis scatter 0.13 deg | 126.70, 0.20 deg |

Per surface the two runs differ by ±0.04 mm RMS except `pocket_wall` (+0.12 mm
corrected) and `rail_6mm_side` (−0.11 mm); `pocket_floor` is unobserved and
`pocket_wall` fails in both. The outer faces' orthogonality deviations are the
one thing that moves consistently, and towards the CAD (east 0.16 vs 0.32 deg,
south 0.70 vs 0.75).

**What this does and does not say.** This gate measures nearest-point distance
from CAD to scan after an ICP placement, and a placement absorbs most of a
0.15-0.55% scale on a 92 mm object: the largest displacement such a scale can
produce at the target's extremes is a quarter of a millimetre, split both ways
about the centre. So a correction of that size is expected to leave this gate
where it was, and it has; the gate confirms the correction breaks nothing, not
that it fixes the size. The 0.8% undersize of the fused platform is a SIZE
measurement (`component-size-bias-results.md`) and is the number the correction
has to be judged on; that comparison has not been run yet.

Two practical notes from running it. The placement search oversubscribes the
machine: with the BLAS and OpenMP pools left free it burned fifteen cores for
more than ten minutes without finishing on this capture, and with
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` it finished placement in under three
minutes (the whole run, registration included, in under four). And `--cache`
writes the fused cloud before placement starts, so a run that is cut off during
placement resumes from the cloud; the cache carries no record of which
deprojection wrote it, which the JSON report now says.
