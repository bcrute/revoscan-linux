# Controlled component repeat — 2026-09-09

## Outcome

The fresh fixed-camera, slow-turntable scan reconstructs successfully without
falling back to general tracking. Its measured X/Y base dimensions differ from
the saved September 4 result by only **0.094/0.101 mm**, but remain **0.660/0.572 mm
short** against the owner's newly supplied mid-height measurements. This is
useful evidence for a repeatable constrained workflow, not a qualified general
handheld fix or proof of a particular calibration fault.

The observed-surface score improves to 0.443 mm RMS, but orthogonality is worse
than the old result and the pocket floor is completely unobserved. No scale
compensation, firmware change or experimental scan mesh was applied.

Evidence directory:
`/home/ben/scans/diagnostics/20260909-component-repeat-Mbsggn/`.
Sealed at 18:30:38 UTC: 1,413 files, 689,631,100 bytes; an independent readback
verified the exact file set and every hash. Manifest SHA-256:
`1a10ba9a067a23004f54a26bba718450a80f3490c826c131f5592e0e16221f00`.
Do not modify this evidence root; later tests and owner clarifications need
new records outside it.
The [previous baseline](component-baseline-results.md) and
[initial readiness check](component-repeat-readiness.md) remain separate records.

## Acquisition and fixed-axis reconstruction

- Original MINI serial `A22704522A6F00A64`; 20,000 us frame time and exposure,
  manual mode 0, gain 24 verified before and after. No settings writes.
- After the operator recentered the target and started the table, 32 depth
  frames confirmed motion and usable framing. The full capture then delivered
  **809/809 complete distinct frames, zero sequence drops and zero short frames,
  4.51660 fps**. Its 180-second duration ceiling ended the run normally before
  the 1,200-frame maximum; `stopped_early` here means the declared duration,
  not a stall or tracking loss.
- Internal CPU/NPU temperatures were monitored before, approximately every
  five seconds during, and after. They rose from 58.5/57.7 to **61.6/60.8 C**,
  the observed maxima. The 80 C test-policy cutoff was never approached. This
  is a three-minute baseline capture, not high-speed thermal-soak qualification.
- The predeclared primary selection is all even-index deliveries: **405
  frames**, matching the old capture's decimate=2, not its duration or physical
  table speed. All 809 raw frames and their hashes/times are preserved.
- Actual delivery monotonic times are passed explicitly to unchanged
  `register_turntable`; raw-file write times are not used. The selected cadence
  is sufficiently uniform that the estimator normalizes it to frame order.
  That logged wording does not mean capture timing was omitted. Project-time
  readback is exact; six timing-selection tests pass.
- The model finds 255.204 selected intervals per turn, about **113.03 seconds
  per revolution** and **1.583 turns** across the scan. Two rounds finish with
  145 pairs about 60 degrees apart, 0.137-degree axis scatter and 0.289 mm final
  centre shift. Median neighbour/quarter-turn/full-turn separation is
  0.072/0.430/0.110 mm. Registration takes 58.88 seconds and preserves
  825,989 points and all 405 poses. No fallback was permitted or used.
- All 25 image-matched revisit checks are checkable; the weakest has 99.050%
  agreement within 1.5 mm. Because the axis/rate model partly enforces closure,
  this is not an independent dimensional-accuracy check.

The full framing audit finds 37 frames whose largest depth component touches
the image edge, at most 14 pixels. Inspected sampled/extreme views retain the
printed target; low-edge contacts include the support surface. This is visual
framing evidence, not semantic segmentation or a claim of complete surfaces.

## Dimensions: updated mid-height calipers

The owner subsequently reported: `81.74, 91.77 measured halfway up the
"platform" of it`. In response to the base-sidewall-versus-bottom-lip question,
this is recorded as **X 91.77 mm, Y 81.74 mm at mid-height of the base platform**.
The original X/Y 91.90/81.85 readings remain preserved. No original print record,
CAD correction, rigid placement, raw frame or reconstructed point was changed.

| Base-plane dimension | New mid-height calipers | September 4 baseline | Fresh repeat | Fresh minus calipers | Fresh minus baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| X (mm) | 91.770 | 91.01569 | 91.10970 | -0.66030 (-0.7195%) | +0.09401 |
| Y (mm) | 81.740 | 81.06712 | 81.16825 | -0.57175 (-0.6995%) | +0.10114 |

Against these updated calipers the old scan is 0.754/0.673 mm short, about
0.822/0.823%. The mid-height clarification accounts for only 0.13/0.11 mm of
the original discrepancy; it does not explain most of the shortfall.

The unchanged helper independently fits opposing base sidewalls at common
mid-height, excluding the first/last 1.25 mm and corners. Repeating both 3 mm
and 5 mm selection windows on both clouds using the new caliper values yields
**identical fitted faces and scan dimensions in all eight checks**. The new
measurements therefore change the comparison, not the recovered geometry.
Six dimension-helper tests and three evaluator SVD equivalence checks pass.

Record locations matter: `dimensions.json` and `closeout.json` retain the
original-caliper comparison; **`midheight-validation.json` is the additive,
updated physical comparison**. The formal uncertainty remains unknown: these
are owner-reported dimensions without instrument resolution or repeated
readings. The two acquisitions also differ in date, framing and table speed,
so a 0.10 mm difference is not a same-setup precision specification.

## Surface scores and missing geometry

The following CAD scores retain the original, predeclared print correction.
They have not been refitted or relabelled as using the new mid-height calipers.

| Quantity | Saved baseline | Fresh repeat |
| --- | ---: | ---: |
| CAD-side observed-point RMS (mm) | 0.50161 | 0.44302 |
| CAD-side observed-point p95 (mm) | 0.89636 | 0.68885 |
| CAD reference points observed | 95.0417% | 95.0667% |
| Judged surfaces under 1.5 mm RMS | 23/24 | 24/24 |
| Worst face orthogonality deviation | 0.74911 degrees | 1.34427 degrees |
| Pocket floor observed | 10.91% | **0%** |
| Pocket wall observed | 67.99% | 50.36% |
| Pocket wall RMS (mm) | 1.60201 | 1.45091 |

The historical surface gate now says pass only for its **24 judged surfaces**.
It excludes the unobserved pocket floor; it is not a whole-target qualification.
Pocket-wall p95 is still 2.905 mm. Height remains unmeasurable without the bottom,
and the 12.1 mm pocket depth is not validated. Another stationary-camera view
from higher up would be needed for the pocket; do not fill it from CAD.

## CAD artifacts, execution and preservation

`measured-target-overlay.FCStd` contains the existing, print-corrected CAD mesh
and measured scan points. No scan mesh was generated. Of the full 825,989
points, 776,520 lie in the same expanded display box used for the baseline;
every sixth of those gives 129,420 displayed points. This is a display crop,
not selection by distance to CAD. Support/outlier points remain visible.

Use `scan-only-axonometric.png`, `overlay-axonometric.png`, and the direct
equal-scale `scan-only-xy-projection.png`. The original FreeCAD
`scan-only-top.png` rendered incompletely and is retained as failed rendering
evidence, not missing-geometry evidence. The direct XY projection of the same
saved points verifies the top outline. Its first plotting attempt found
matplotlib unavailable; existing Qt produced it without installing dependencies.

FreeCAD's saved document passes ZIP integrity validation. Its task-created
process did not exit on the scripted quit; after validation only that exact,
command-checked process was terminated with SIGTERM. No pre-existing CAD
process or document was touched.

The cached CAD evaluation completed in 162.75 seconds, with no timeout. It is
the same process-local reduced-SVD evaluator used for the baseline, with only
its input capture path changed; the dimension helper is byte-identical. The
largest face call is 73,894-by-3. This is still diagnostic-only, not a promotion
of an optimization into frozen production sources. Legacy report wording that
rigid registration absorbs scale remains incorrect and is not endorsed here.

Closeout verified all 94 source/reference files and snapshots, all original and
selected new raw hashes, every one of the old sealed baseline's 122 manifest
files, and all 195 old capture files with their original modification times.
M27 remains suspended. No scanner/firmware/calibration/dependency or production
source changes were made. Capture, registration, evaluation, tests, images and
the new owner measurements remain in the evidence bundle.

## Next decision

The narrow fixed-camera/slow-turntable path has now recovered comparable
external dimensions on an independent capture. General handheld tracking is
still unqualified. The approximately 0.7-0.8% residual shortfall in both axes is
a repeatable clue, not proof of uniform depth-scale error: lateral calibration,
depth calibration, pose estimation and remaining measurement details are not
separated by these two results. Do not apply an approximately 1% compensation.

The next useful work is to isolate that size bias and agree the required
layout/clearance tolerance before accepting scans for chassis fit. A second
independently dimensioned span/object would test generalization; a higher view
would address pocket coverage but would not by itself validate scale. Neither
requires resuming the broad tracker matrix or pushing FPS/firmware changes.
