# Can spatial support and residual-tail influence — 2026-09-10

Subsequent checkpoint: the [stationary physical-reference readiness review](stationary-reference-readiness.md)
is complete. Existing reviewed views do not yet supply a qualified independent
lateral reference; the prospective pocket-block outline fails two perimeter
support checks. The next dependency is a suitable measured parallel-plane step.
The can result below and its sealed evidence remain the earlier checkpoint.

## Outcome

This saved-data diagnostic closes two narrow questions, not the dimensional
acceptance gate:

- The observed missing-pixel patterns and original selected-pixel sets recover
  an ideal 58 mm cylinder to within **0.000213 mm** in all 14 historically usable
  cases. This is a numerical control with noiseless synthetic depths, not real
  scanner accuracy. Missing support alone does not explain their real size error
  under the tested model.
- One fixed gross-residual-tail removal/refit makes the bare-can diameters
  generally **smaller**, not closer to the physical reference. The sprayed-far
  mean changes only **-0.003481 mm**, from 57.716585 to **57.713105 mm**. It remains
  0.286895 mm short, about **0.49465%**, outside the owner's below-0.3% limit.
- All 16 real refits converge, all point-estimate diameters remain outside the
  0.174 mm limit, and all preserved-style validation verdicts remain
  **inconclusive**. The two historically unusable captures stay unusable.

No filter, scale correction or calibration change is promoted. No scanner,
firmware, production source, preset, mesh or M27 campaign is changed. This result
does not identify a hardware/firmware fault or eliminate real noise interacting
with sampling, target nonideality, optical effects, or calibration error.

Evidence root: `/home/ben/scans/diagnostics/20260910-can-support-NKJ6NM/`.
Sealed at 2026-09-10 15:19:59 UTC: **236 files / 75,462,705 bytes**, excluding
the manifest itself; manifest SHA-256
`17bd8796d51c6ee65516200f4d093f2def3061dfba9fbb81f8b0b56b9f197d55`.
Exact file-set/size/hash verification passes. Do not modify this root. Archived
document snapshots precede these seal details. Closeout also confirms all 78
live snapshot source files and all 449 frozen live/saved M27-context files are
unchanged; PID 2009261 remains stopped.

Inputs are the immutable [original replay](can-replay-results.md) root, including
the exact recovered procedure, current source snapshot, 16 raw frames and cached
fits. Input identities, every original selected-point/pixel hash and counts are
verified. The existing baseline fits are reused without redundant optimization.

## Declared experiment and audit-driven amendment

`PROTOCOL.md` was written before execution. For each case it specified:

1. An ideal radius-29 mm cylinder at the real baseline's fitted centre/axis,
   rendered with explicit factory intrinsics/depth scale and uint16 rounding.
   The synthetic diameter is declared truth, not a fitted scale for real data.
2. The identical ideal depth codes masked by that real frame's nonzero pixels.
   Both full and masked frames undergo the recovered per-case Z band, stride,
   initial fit and, where applicable, axial crop.
3. One real-data refit after retaining signed radial residuals within
   `max(6 * 1.4826 * median(abs(residual - median(residual))), 0.05 mm)` of their
   baseline median. The selection is inclusive, applied once, with no parameter
   sweep, recropping or reference-dependent tuning.

The ideal cylinder uses half-length 1000 mm only to keep caps out of a body-only
control; it is not a claim about the physical can height. All generated points
are within about 64 mm axially of their model centre; no depth code is beyond
range. Independent radial rescoring confirms the ideal raw points differ from
the declared surface only within the quantization bound. All real points are
retained in evidence and scored against the tail refit, including removed ones.

The original audit exposed an important distinction: recomputing a percentile
band and every-fifth-point stride can change the exact final pixel subset, even
when the original selected pixels all intersect the ideal body. Far-case final
overlap was only **4.10–57.45%**; near-case overlap was about 98.9–99.3%. The
raw-mask procedural control therefore was not called an exact-support control.

`EXACT-SUPPORT-ADDENDUM.md` was then declared before a single additional bounded
control per case. It reused each existing full ideal frame at the **exact
original selected pixels, in the same order**, without new rendering, filtering
or real-data refitting. Any missing original pixel required an explicit refusal,
not a fit to a reduced subset. Original protocol/scripts/results remain intact.
This is a documented audit correction, not an undisclosed original test or an
automatic parameter search.

## Synthetic support results and negative controls

For all 14 historically usable cases, 100% of the real originally selected
pixels intersect the ideal body. Across those cases:

| Control | Maximum absolute diameter error |
| --- | ---: |
| Full ideal body + historical selection procedure | 0.000138 mm |
| Real nonzero-pixel mask + historical selection procedure | 0.000210 mm |
| Exact original selected pixels and order | 0.000213 mm |

The maximum masked-minus-full procedural diameter change is 0.000218 mm. All
these fits converge and satisfy the existing arc-conditioning check. These very
small numbers are synthetic arithmetic errors; they do not estimate uncertainty
on the real object or validate factory calibration.

The rejected controls remain explicit:

- **Sprayed 95:** the masked procedural fit is 0.016744 mm short and fails the
  conditioning check (arc half-angle 0.298707, floor 0.35). Using the exact
  original 415 selected pixels instead gives a converged, well-conditioned
  ideal fit 0.003539 mm short. This demonstrates why the two selection controls
  must be distinguished; it does not rescue the real 29.374 mm baseline or
  42.487 mm tail-refit estimates.
- **Sprayed 125:** only 91.9216% of its 8,566 original selected pixels intersect
  the ideal body. Its procedural ideal fit looks numerically good but is not
  accepted as closely matched support. The exact-support control **refuses**
  because 692 selected pixels have no model intersection; all missing indices
  are preserved. No replacement pose or smaller point subset is tried.

## Real tail influence

All rows use the fixed one-pass rule. Means are over retained frames only, not
missing historical acquisitions; hub/direct near frames are kept separate.
Distance-like labels are historical scanner-derived labels, not independently
measured standoffs. Rounded numbers below are not extra physical precision.

| Retained group | n | Original mean diameter | Tail-refit mean | Mean change | Mean points removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bare 98, hub | 2 | 56.640545 | 56.310461 | -0.330084 mm | 1.1921% |
| Bare 98, direct | 3 | 56.663976 | 56.361452 | -0.302523 mm | 1.1394% |
| Bare 120 | 3 | 57.704697 | 57.458427 | -0.246270 mm | 0.4833% |
| Bare 244 | 3 | 57.441476 | 57.416993 | -0.024483 mm | 0.4616% |
| Sprayed 242 | 3 | 57.716585 | 57.713105 | -0.003481 mm | 0.0471% |
| Rejected sprayed 95 | 1 | 29.374161 | 42.487421 | +13.113259 mm | 2.1687% |
| Rejected sprayed 125 | 1 | 61.933504 | 58.214807 | -3.718697 mm | 2.2297% |

For bare-120 frame 1, 445 of 98,275 points (0.4528%) account for 85.17% of the
baseline squared radial residual. Removing them drops kept-point RMS from the
original whole-set 0.21333 mm to 0.07907 mm, but diameter moves from 57.67971 to
57.45832 mm, further below 58 mm. Rescoring **all** original points gives
0.21459 mm RMS: the removed points have not ceased to exist. Separate baseline
RMS on kept points is also recorded, avoiding an unmarked support comparison.

The three bare-120 tail-refit diameters span only about 0.001185 mm, much tighter
than the original 0.043920 mm, while their mean physical error worsens to
0.541573 mm. Thus cleaner surfaces and tighter same-recording repeatability can
coexist with poorer external dimensional accuracy. Neither is a calibration test.

The sprayed-far cases lose only 4, 8 and 1 points, respectively. Their diameter
changes are -0.002364, -0.008550 and +0.000472 mm. Their broad undersizing is not
explained by these gross tails. Residual structure remains: the fitter's
correlation indicator is about +0.124 to +0.144 after trimming, above its 0.12
threshold. This indicator describes structure, not its cause. On bare-120 it
becomes stronger in magnitude, about -0.470 to -0.592.

Fixed 8-by-5 image-tile statistics and per-point residuals/masks preserve the
spatial evidence. For example, bare-120 frame 1 has sizeable, oppositely signed
median residuals in tiles with no removed points; the data are not described
solely by the handful of extreme returns. A residual-based exclusion is not
proof of invalid depth: genuine surface departures and correlated optical or
geometric error can also create those patterns.

## What changes next

The host-side gross-tail mechanism is real, but its tested correction does not
solve the current accuracy problem. Stop this can-filter investigation without
promoting it. The residual size error now needs a reference/view-dependence
check, not a guessed calibration coefficient inferred from one cylinder.

The next bounded question is whether a fixed calibration correction could
transfer across distance and image position on **independently observed,
physically measured geometry**. First inspect the existing stationary printed
target evidence for that comparison. The lateral references must be the actual
91.77/81.74 mm mid-height sections, not a top outline at a different height,
uncorrected print CAD, or dimensions imposed by a shared turntable model.
Record reference uncertainty and which opposing datums are truly visible.

If the saved data cannot supply independent lateral observations, name that
specific gap before asking for another capture. The needed physical reference
is a caliper-measured pair of datums whose separation can actually be observed
in the chosen view(s), with the measurement section and uncertainty recorded.
A generic extra can scan, unknown coating thickness, or an unobserved rear face
cannot supply it. Freeze any candidate correction on a designated calibration
subset and assess different views/ranges and a different component held out;
never rescale each scan to its own calipers. No firmware necessity follows here.

All [four component acceptance gates](component-acceptance.md) retain their prior
status. This diagnostic supplies no new independent acquisition or CAD delivery.

## Verification and preservation

Fourteen wrapper tests pass, including identity subtests on all 16 real frames.
Three additional ordering/missing-pixel-refusal tests pass: **17 distinct tests**,
not counting repeated data audit assertions as extra tests.

The initial 16 workers complete in **97.6519 seconds** with 32 procedural ideal
controls and 16 real refits. The exact-support addendum's 16 workers complete in
**32.9783 seconds**, containing 15 ideal fits and one deliberate support refusal.
No worker process fails or times out; the refusal is a recorded geometric
outcome, not hidden as a process failure or an accepted scan. No numerical retry.

`support-audit.json` and `exact-support-audit.json` verify point identities,
row/pixel order, exact masks, all artifact hashes, quantization bounds, original
and refit residual scores, source imports and every outcome. Original scripts
and the addendum are separately hash-bound before their respective executions.
The parent root's exact 306-file set, byte counts and hashes remain unchanged.
Per-case binary point records describe their own format and retain all points,
original/refit residuals and keep flags. Both synthetic raw images, selected
indices, all fit results, refusals, tests, execution records and source bindings
are preserved. The scanner was not activated or queried for temperature.
