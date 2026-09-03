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

## `outer_west` at 0% is not explained, and one explanation was wrong

The sweep is roughly 1.5 revolutions, so every face passed the camera. Two
explanations have been tested and neither survived.

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

**Untested candidates**, in rough order of plausibility:

- **Projector shadowing.** The projector and camera are separate apertures, so
  geometry visible to the camera can be unlit. The target's ramp rises to 56 mm
  and could shadow a side face at the rotations where that face turns toward the
  camera. This would be asymmetric between faces, which matches `outer_east` at
  86% against `outer_west` at 0%.
- **Grazing incidence** on a near-vertical face, which returns little from a
  structured-light sensor. Does not by itself explain the east/west asymmetry.
- **Something in the capture geometry** specific to this session's placement.

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
