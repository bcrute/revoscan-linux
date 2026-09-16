# Component size-bias localization — 2026-09-09

## Outcome

The remaining undersizing is not explained by point thinning, the 60-300 mm
depth band, an accidentally scaled pose/CAD transform, or the plane-fit formula
checked here. Keeping every raw point changes base widths by less than 0.008 mm,
against the fresh scan's 0.660/0.572 mm shortfalls.

A separate raw-plane centre calculation changes the estimated widths materially,
even with the original axis direction retained. Its result also depends on axis
and wall assumptions. This demonstrates estimator sensitivity, not a validated
tracking correction or proof of faulty hardware. No geometry or calibration was
changed. A new stationary physical view is the next bounded checkpoint.

Evidence root:
`/home/ben/scans/diagnostics/20260909-component-size-bias-wOfT9E/`.
Sealed at 19:11:12 UTC: **215 files, 83,420,374 bytes**. Independent readback
verified the exact file set and every hash. Manifest SHA-256:
`b19d2dcde70c095f0946f068c8426469d13bfaea24255c7f9e1b31cf1e619510`.
Do not modify the evidence root. Its document snapshots precede this seal
metadata; later physical work must use a new root.
The [baseline](component-baseline-results.md) and
[fresh repeat](component-repeat-results.md) remain separate sealed inputs.
All analysis here is offline: no scanner access or new temperature measurement,
setting/firmware/calibration write, production-source change or experimental mesh.
M27 remains suspended; its unfinished matrix was not resumed.

## Fixed-pose controls

Use all 190 baseline frames and all 405 predeclared fresh comparison frames,
with their original poses and CAD placements frozen. X/Y comparison values are
the owner's base-platform mid-height calipers, 91.77/81.74 mm. The previous print
correction still supplies ROI heights/corner exclusions, not a new scale fit.

Every prepared point in both saved clouds reproduces exactly, in order; every
saved CAD coordinate also reproduces exactly. All 597 pose/placement rotations
are proper rigid transforms: maximum orthogonality defect 1.78e-15 and singular
value error 1.11e-15. These mappings cannot explain a hidden 0.7% scale change.

| Capture | Point treatment, fixed original poses | X (mm) | Y (mm) |
| --- | --- | ---: | ---: |
| Baseline | Original prepared cloud | 91.015687 | 81.067115 |
| Baseline | All raw points, same depth band | 91.023082 | 81.072477 |
| Fresh | Original prepared cloud | 91.109701 | 81.168254 |
| Fresh | All raw points, same depth band | 91.113246 | 81.170758 |

The maximum change is 0.007395 mm. Removing the depth band while keeping the
same bounded base-sidewall ROI changes neither width: zero relevant ROI points
lie outside the band in either capture. The controls examine 15,676,184 and
23,344,592 nonzero raw points respectively, retaining 943,808 and 1,532,392 base
ROI points. They do not widen the registration band or change any pose.

Three/five-millimetre selection windows agree to far below 0.001 mm. Independent
covariance/TLS plane fits change the corresponding coordinate-regression widths
by less than 0.00004 mm. This excludes these particular handling/measurement
choices as the main cause on these inputs, not every possible preprocessing
effect on newly estimated poses. Per-frame fits and support are saved; their
view-dependent coordinate changes still include the frozen estimated poses.

## Raw-plane centre diagnostic and axis sensitivity

Fit raw, untransformed camera-coordinate planes for visible base walls. Original
poses/CAD are used only to label walls and select a broad mid-height ROI. Require
at least 200 retained points and broad transverse/vertical support per plane.
There are 270 accepted baseline plane observations and 556 fresh observations
per ROI window, with all refusals preserved.

For a rotating near-prismatic base, solve the common spindle centre and four
free wall offsets from projected plane equations `d_i = n_i · c + h_face`.
Opposite offsets give diagnostic X/Y support-line widths. Caliper values do not
enter this solve. No old frame angles, trajectory translations or nominal widths
are fitted as truth. The three axis controls are:

- Uncentred: infer axis from the wall-normal covariance, assuming vertical walls.
- Within-face: infer axis from each wall's normal variation, allowing fixed taper.
- Original axis: retain only the old trajectory's rotation-derived axis direction;
  independently solve centre and wall offsets. This mode is explicitly not fully
  independent of the original trajectory.

All observations get equal weight in the ordinary fit. Also report a fixed
five-round, three-MAD residual trim with a 0.02 mm threshold floor. Do not choose
the result nearest the calipers. The table uses the 3 mm ROI; 5 mm results are
identical for fresh and within 0.00012 mm for baseline.

| Capture | Axis model | Ordinary X / Y (mm) | Trimmed X / Y (mm) |
| --- | --- | --- | --- |
| Baseline | Uncentred | 91.220635 / 81.272839 | 91.169738 / 81.219414 |
| Baseline | Within-face | 91.142455 / 81.185751 | 91.112100 / 81.149069 |
| Baseline | Original axis | 91.215013 / 81.265863 | 91.163272 / 81.211727 |
| Fresh | Uncentred | 91.492923 / 81.541224 | 91.430332 / 81.479306 |
| Fresh | Within-face | 91.327148 / 81.355899 | 91.308532 / 81.337446 |
| Fresh | Original axis | 91.462214 / 81.506681 | 91.405759 / 81.451201 |

The fixed-original-axis trimmed fresh widths are about 0.296/0.283 mm larger
than the saved cloud's dimensions without fitting a scale or imposing calipers.
The uncentred and within-face axis estimates differ from the old trajectory's
axis by 0.823/1.910 degrees for baseline and 1.036/2.339 degrees for fresh.
The small raw-plane linear-fit residuals, about 0.022-0.034 mm, are **not**
dimensional uncertainty or external accuracy. Noise and view-dependent bias
in normals/offsets can also move the fitted centre and dimensions.

All default-axis numerical results reproduce the initial diagnostic exactly.
Synthetic controls recover freely chosen widths, including tapered walls and
arbitrary rigid coordinate changes, and reject missing/degenerate support.
They test the implementation's stated geometry, not whether the real print and
raw measurements satisfy it. The support-line model, common cross-section,
wall segmentation and observation weighting differ from fitting aggregated
cloud faces. Its change cannot be attributed solely to correcting pose error.
The estimates are not interchangeable caliper readings or confidence bounds.

## Calibration continuity and historical evidence

The unchanged software intrinsics exactly match this unit's previously saved
`Pl.bin`: fx=fy=898.6829833984375, cx=42.43747329711914,
cy=192.8326873779297, 640x400. Both saved scale denominators are 200;
`DEPTH_SCALE_METRES` is exactly float32(1/200). Agreement with these saved
numbers is not physical optical-calibration validation or a new device read.

The [calibration-semantics review](reliability-calibration-semantics.md) already
records the YAML serial mismatch and distinguishes raw-camera from rectified
intrinsics. It does not justify substituting raw camera matrices, recentering
the principal point or adding distortion correction to rectified depth. There
is no new units/convention or simple transcription fault in this continuity check.

Older single-frame cylinder measurements remain context independent of tracking:
the recorded 58.0 mm body had both near-correct and materially biased estimates
depending on view/range/surface treatment. The current review did not recover
the exact historical thinning/cropping harness, and therefore did not rerun or
claim reproduction of those can measurements. Historical rejected sprayed-can
fits remain rejected; old principal-point/distance-curve causal claims were
withdrawn by later controls. See the metric manifest and calibration review,
not an isolated early headline.

The present two captures do not uniquely identify a depth-scale, depth-offset,
lateral-calibration, raw-distortion or pose-model fault. A range-dependent error
or additive depth offset is a possible future hypothesis, not an observed
mechanism or grounds for a correction. Do not fit more compensation parameters
against the same two caliper spans and call that validation.

## Execution, tests and preservation

The first fixed-pose runner failed before publishing dimensions because it
converted `Rigid.apply_all`'s iterator directly to a NumPy object array. Its
exact script, preflight and error explanation remain preserved. An additive v2
materializes the iterator, leaves the math/input choices unchanged, and finishes
the full sampling comparison in 41.07 seconds. Both attempts had an external
600-second limit; the successful run was not near it.

The initial helper tests passed 8 cases, raw-plane tests 9, and axis-sensitivity
tests 4. Axis tests were repeated after making fixed-axis provenance explicit;
the repeated report does not add four distinct tests. A final persistent
iterator regression is included in the combined verification. These are narrow
diagnostic software tests, not full-repository or scanner qualification.
The final combined run passes all **22 distinct tests**, zero failures, errors
or skips. Repository `git diff --check` also reports no whitespace errors.

Protocols, scripts, source snapshots, point ROIs/frame identities, raw-plane
observations/refusals, full numerical results and execution checks remain in the
evidence root. No alternate centre/axis model has replaced either saved cloud,
FreeCAD document, reconstruction or calibration.

Finalization at 19:10:49 UTC reverified 82 sampling sources and both snapshots,
609 input hashes/times, the fresh capture's 94 sources/times/snapshots, and all
449 frozen live/context files plus the separate 182-file execution snapshot.
The sealed baseline's 122 files and fresh repeat's 1,413 files have exact file
lists and hashes, with both prior manifest hashes unchanged. PID 2009261 is
still command-verified stopped. The first closeout attempt used the wrong
historical snapshot directory; its failed script and correction explanation
are retained. The corrected audit uses the original freeze script's declared
directories, not a waived integrity check.

## Next operator action and stopping condition

Stop the table and provide a steep downward, supported stationary view into the
small rectangular pocket, exposing both the floor and surrounding plateau top.
The assistant will check framing and collect a short stationary sequence after
the operator reports ready. No hand-held sweep or repeat of all caliper readings
is needed for that first step. The detailed bounded procedure is preserved as
`NEXT-STATIONARY-CHECK.md` in the evidence root.

Measure plateau-to-floor separation in individual raw camera-coordinate frames,
without inter-frame alignment or CAD scale fitting, against the existing
12.10 mm physical reading. This also addresses the fresh scan's zero pocket-floor
coverage. It is one local depth check, **not** a complete scale discriminator:
0.7% of 12.10 mm is only 0.0847 mm, and measurement uncertainty may exceed that;
good depth separation would not establish correct lateral width calibration.

If the new view cannot discriminate the remaining error, request a specifically
measured larger visible span/step instead of another broad fitting sweep. Agree
actual chassis clearance tolerance and sufficient component coverage before
acceptance. General freehand tracking remains unqualified. No new scan has run
at this checkpoint; missing physical visibility is the next concrete dependency.
