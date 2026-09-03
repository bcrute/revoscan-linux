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
0.48 at 1.0 mm, a ratio of 0.48-0.53 throughout - which is the signature of a
shell filled with scenery rather than a surface. An orientation filter, which
fixed exactly this problem for the orthogonality gate, does **not** fix it here:
the worst contaminant is coplanar with the surface it contaminates, because a
turntable top is horizontal and so is `base_top`.

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
