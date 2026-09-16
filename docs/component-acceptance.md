# Component-scanning acceptance checklist

Agreed working finish line, 2026-09-10: the owner accepted the four criteria
below. Test results choose the next diagnostic; they do not relax these criteria.
The deliverable is useful component geometry for a 1/10-scale crawler chassis
in the existing Linux/CAD workflow. This is not a claim of an industry-wide
0.3% specification or of accuracy already achieved.

## Fixed acceptance criteria

1. **Dimensional accuracy:** absolute relative error below 0.003 on each
   preselected, physically measured, observable dimension. Compare the scan to
   the actual object, not uncorrected nominal print CAD. Do not average away a
   failing dimension or rescale each scan to its own reference measurements.
2. **Repeatability and transfer:** three independent acquisitions of the same
   object meet those limits with a fixed acquisition/processing recipe, followed
   by a check on a different component. Reprocessing one recording, neighboring
   frames, and three stages of a single experiment are not three independent
   workflow qualifications. Freeze any calibration before the held-out check.
3. **Useful observed geometry:** the component's relevant exterior and mounting
   references are observed sufficiently to position it and check the intended
   clearances in CAD. Declare the required surfaces/features before qualification.
   Hidden, filled or CAD-inferred geometry is not a measurement. An aggregate
   coverage percentage cannot substitute for a missing critical feature.
4. **Usable Linux workflow:** capture, stop, save, reopen and export without
   unexplained reconnects or repeated parameter tuning. Initially a supported
   scanner and slow turntable are acceptable. Check units and dimensions after
   CAD import, not only in an intermediate cloud.

Reference uncertainty and measurement locations must be recorded. If their
uncertainty prevents resolving a boundary, mark that comparison inconclusive;
extra printed decimal places are not extra evidence. The 0.3% dimension rule
does not independently qualify a thread, press fit, hole position or every
surface point. Critical interface checks must name their dimension/datum and
actual assembly requirement before they are scored.

The current printed platform's mid-height references are X 91.77 mm and
Y 81.74 mm; the corresponding error limits are 0.27531 and 0.24522 mm. These
are distinct from the earlier overall-print readings retained in the physical
reference. For the historical 58.0 mm can the limit is 0.174 mm.

## Current scorecard

| Criterion | Current evidence | Status |
| --- | --- | --- |
| Dimensions | Latest combined turntable control is X/Y 0.74996/0.63347 mm short, about 0.8172%/0.7750%. Replayed retained sprayed-can mean is 0.283415 mm short (0.4886%); fixed tail-refit diagnostic is 0.286895 mm short (0.49465%). All real can comparisons remain inconclusive under the preserved validation rules. | Not met |
| Repeatability / different component | Existing repeats show common undersizing; no three passing independent acquisitions and passing different-component check exist. | Not met |
| Required geometry | Pocket-floor visibility is pose/exposure-dependent; latest short-exposure turn has no scored floor coverage and upper-ramp margin is limited. | Not qualified |
| End-to-end workflow | Latest long acquisition and restoration are clean, but that does not complete three dimension-qualified CAD deliveries. | Partially demonstrated |

Sources: [full-turn comparison](component-turn-exposure.md),
[can measurements](metric-validation.md),
[calibration chronology](reliability-calibration-semantics.md), and
[practical component history](practical-component-scanning.md).

## Rules for the next investigation

Record the question, the decision it changes, its fixed inputs/method, and its
stopping point before running it. Retain refusals and failed controls. A negative
result is useful when it closes an explanation or identifies a specific missing
observation; it is not permission for an automatic parameter matrix.

The immediate bounded task is to recover/replay the August 24 can measurement
procedure from the preserved evidence, or identify exactly why it cannot be
reproduced. Establish what is supported about dimensional error before tracking.
An explicitly new replay method must not be labeled the historical method just
because it matches a rounded diameter. No new capture is initially needed.

Completed checkpoint, 2026-09-10: [the original method was recovered and all 16
retained frames reproduced](can-replay-results.md). Current/historical source
trees give identical geometric results. Original summary denominators and
inconclusive verdicts are now explicit. This closes the recovery question, not
the dimensional acceptance gate. The next selected diagnostic is saved-data
measurement sensitivity to spatial support/outlying returns.

That [support/tail diagnostic is now complete](can-support-results.md). All 14
historically usable exact-pixel ideal controls recover the declared diameter
within 0.000213 mm, but the fixed real-tail refit does not solve undersizing.
The sprayed-far mean moves only -0.003481 mm; bare-can means generally shrink
further. Both rejected captures retain their status. All 16 real point estimates
remain outside the limit and all comparisons remain inconclusive. No filter or
calibration is promoted. A documented exact-pixel addendum closes the difference
between raw support and percentile/stride-selected support; 17 tests pass.

Next selected question: readiness of an independent physical lateral-reference
comparison across view/range, starting with existing stationary printed-target
evidence. Match the actual mid-height measurement datums and reference
uncertainty; do not substitute hidden faces, a different-height outline, or
dimensions imposed by a shared pose model. Identify any missing physical
observation before requesting a new capture.

That [stationary-reference readiness check is now complete](stationary-reference-readiness.md).
The existing mid-height base widths still require opposing views/pose or an
independently validated edge method. A prospective pocket-block outline is
not adopted: all 120 interior planes are supported, but two perimeter guides
lack outward observed-surface support in 99.3–100% of transects per stage. No
physical width or calibration was fitted; the 12 software tests are not a new
accuracy pass. The old two-view pocket data also are not a controlled range sweep.

Earlier physical dependency: the owner was asked for a suitable rigid,
matte, flat/parallel-faced block about 40–80 mm thick to measure and place flush
on a flat matte support. That enables a larger directly observed plane-spacing
check, with contact, reference uncertainty and later directional/range checks
explicit. A passing step height alone would not qualify lateral dimensions or
the complete workflow.

Current update, 2026-09-10: a new block would require printing, so its necessity
is not assumed. The owner supplied underside-to-top heights **16.08/25.98 mm**
for the two highest treads of the existing staircase, not the pocket plateau.
The [supplemental record](../reference/targets/reconstruction_v1/prints/physical-print-20260831-steps-20260910.json)
preserves both readings and their datum. Their provisional difference is
**9.90 mm**; do not subtract nominal base thickness or replace a different
reference dimension. Next is a supported, stopped view exposing both tread tops,
then a correspondence/support check. Repeated physical readings and reference
uncertainty remain unresolved. See the [readiness update](stationary-reference-readiness.md).
No new capture, fit or acceptance pass accompanies these owner measurements.

Subsequent [first staircase capture](component-staircase-results.md) now executes
that setup: 120 settled frames, zero delivery drops/shorts, original settings
restored, maximum CPU/NPU 59.8/59.1 C. All 480 fixed patch-pair fits have support,
but primary spacing is about 10.243 mm versus provisional 9.90 mm, with
nonparallel planes and evaluation-location/direction dependence. Exposure does
not resolve the discrepancy. No accuracy gate passes; next is a more downward
stopped view broadening both tread interiors. No new print is required for it.

That [second, more downward view](component-staircase-downward-results.md) now
completes cleanly. Primary spacing is 9.931477 mm (+0.031477 mm / +0.3179% from
provisional 9.90 mm); patch/direction dependence and unknown reference uncertainty
make accuracy inconclusive, not a near-boundary pass. The fixed-pixel control
confirms local-shape improvement at 10 ms without a changing return mask. Next
is repeated, location-identified underside-to-top calipers and instrument
information, keeping the scanner angle. No global preset/calibration or gate
is promoted, and no new print or immediate scanner reposition is requested.

This checklist does not restart firmware/FPS research, the suspended M27 batch,
general handheld qualification, or the experimental mesh. Firmware remains a
possible later response to an isolated device-side issue, not an acceptance
milestone. Earlier sealed evidence stays read-only; new experiments get new roots.
