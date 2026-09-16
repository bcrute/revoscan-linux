# Practical objective: component geometry for chassis layout

Current pause point: [2026-09-10 scanner checkpoint](scanner-checkpoint-20260910.md).
The owner requested an accurate record before seeking a second opinion. The
checkpoint separates completed tests from pending caliper readings, current host
observations from last verified device state, and findings from hypotheses. No
new experiment or production change accompanies it.

Latest result, 2026-09-10: [more downward staircase view](component-staircase-downward-results.md)
completes cleanly and puts primary spacing at about 9.931 mm versus provisional
9.90 mm. At fixed pixels, 10 ms materially improves local shape; residual
patch/direction and physical-reference uncertainty still prevent qualification.
Next is repeated underside-to-top calipers at identified tread positions plus
instrument information. Keep the present scanner angle; no new print or
immediate reposition is requested. Original settings restored; no calibration,
production or acceptance change. Earlier next actions below are historical.

Latest executed check, 2026-09-10: [first staircase view](component-staircase-results.md)
captures cleanly and restores the original settings (CPU/NPU maximum 59.8/59.1 C).
Both tread tops return depth, but about 10.243 mm versus provisional 9.90 mm is
not a qualified height: the narrow-patch planes differ in angle and evaluation
direction/location changes the result. The exposure bracket does not resolve
that. Next is a more downward, supported/stopped view with broader tread
interiors on the same object. No new print, calibration or accuracy pass.

Latest owner update, 2026-09-10: [reuse the existing staircase before requiring
a new print](stationary-reference-readiness.md). Its two highest tread tops are
reported **16.08/25.98 mm from the underside**, giving a provisional **9.90 mm**
difference. These do not measure the separate pocket plateau. The original
print/CAD adjustment is unchanged; the [supplemental record](../reference/targets/reconstruction_v1/prints/physical-print-20260831-steps-20260910.json)
retains the readings and limitations. Next setup is a supported scanner and
stopped table with both staircase tread tops visible, not the pocket floor or
underside. No new capture or accuracy result; earlier next-action wording below
is historical.

Latest checkpoint, 2026-09-10: [stationary-reference readiness](stationary-reference-readiness.md)
identifies a physical-reference gap rather than a calibration fix. The measured
base mid-height widths are not independent same-view opposing-plane dimensions.
A direct pocket-block outline candidate was checked in all 120 existing 10 ms
frames: every interior plane has support, but two perimeter guides lack outside
observed-surface support in 99.3–100% of transects per stage. Missing-depth borders
are not fitted into a precise width. Twelve tests and the full sample audit pass;
no scanner, software recipe or calibration change. The owner has been asked
whether a rigid matte 40–80 mm thick block/box with parallel faces is available
for a caliper-bound, flush-supported plane-spacing check. Its contact, uncertainty
and later tilted/range checks matter; one height result would not qualify X/Y.

Latest offline checkpoint, 2026-09-10: [can support/tail influence](can-support-results.md)
does not produce an accuracy fix. Ideal depths at the exact original selected
pixels recover the known diameter within 0.000213 mm in all 14 historically
usable cases; the real sprayed-far mean moves from 57.716585 to 57.713105 mm
after one fixed tail removal, still 0.49465% short. Bare-can means mostly get
smaller despite cleaner/tighter fits. All real comparisons remain inconclusive,
the two rejected captures remain rejected, and no processing/calibration change
is promoted. Seventeen tests pass; both bounded execution phases and all
point-level evidence are preserved. Next is checking whether existing stationary
target evidence supplies an independent, physically measured lateral-reference
comparison across view/range, with exact datum/height visibility and uncertainty.
Earlier dated entries below preserve their checkpoint-specific next actions.

Latest offline checkpoint, 2026-09-10: [can-procedure recovery/replay](can-replay-results.md)
recovers the original commands and exactly reproduces all 16 retained per-frame
diameters to their historical reported precision. Current/historical source
trees produce identical geometry; all fits converge but all comparisons remain
inconclusive under the preserved validation rules. The retained sprayed-far mean
is 57.716585 mm against 58.0 mm (-0.4886%). No acceptance criterion is newly
passed, no capture is needed to repeat that finding, and no firmware necessity
is established. Next selected question: sensitivity of the single-frame
measurement to real spatial support/outlying returns, using saved data first.

Current acceptance decision, 2026-09-10: the owner accepted the
[fixed component acceptance checklist](component-acceptance.md): below 0.3%
on preselected physical dimensions, three passing independent acquisitions with
a fixed recipe and a different-component check, useful observed CAD geometry,
and a repeatable Linux capture/save/export workflow. These are qualification
goals, not completed capabilities. The immediate offline task is recovery/replay
of the existing August 24 can measurements; that experiment was already done
and must not be proposed as missing. The earlier dated updates below remain
history, including tolerance questions and superseded next actions.

Latest turn-test update, 2026-09-10 UTC: the
[20/10/20 ms full-turn bracket](component-turn-exposure.md) acquires 2,470 full
deliveries/2,430 settled frames without drops or transport faults. Peak CPU/NPU
63.8/63.0 C, original settings restored. The ordinary 10 ms reconstruction is
about 3 mm undersized despite cleaner stopped surfaces: its four-round solver
returns while the centre update is still 6.278 mm, above the existing 0.5 mm
convergence criterion. The two 20 ms fits settle below 0.2 mm and retain roughly
0.54–0.83 mm size shortfalls. One targeted, saved-data continuation reaches
convergence at round six, but remains about 1.0–1.1 mm undersized. An explicit
equal-weight fixture estimate from both 20 ms brackets, applied to short points
only, yields X/Y 91.02004/81.10653 mm: within 0.003 mm of the baseline average,
still 0.74996/0.63347 mm short. This localizes the added short-exposure size
error to fixture estimation in this case; it does not solve the shared bias.
The owner has been asked the acceptable dimensional tolerance. No universal
10 ms preset, production/firmware/calibration change, mesh or M27 resumption.

Latest full-view update, 2026-09-09: the
[genuinely stopped retry](component-fullview-exposure.md) supports a real local
surface-quality improvement at 10 ms: four fixed planar patches retain their
points and have 1.6–4.6 times lower full-input plane residuals, reverting at the
final 20 ms bracket. A fifth selected patch refuses every plane fit. Whole-image
losses occur around the turntable AND object edges, so no blanket coverage or
absolute-accuracy claim follows. Acquisition is clean: 160 deliveries, zero
drops/shorts, 4.524 fps, maximum CPU/NPU 60.2/59.5 C, original settings restored.
Next: slightly more lower framing margin and a slow turntable, then the prepared
full-rotation 20/10/20 ms comparison against existing X/Y calipers. It has not
run; no production/default/calibration change or M27 resumption.

Previous full-view update, 2026-09-09: the
[first whole-object bracket](component-fullview-exposure.md) completed with 160
full deliveries, zero drops/shorts, temperatures at most 61.6/60.8 C and clean
baseline restoration. Raw frames show continuous object rotation, so the
intended stationary exposure comparison is refused; no quality or dimension
result is accepted. A lower corner also reaches the image bottom during the
turn. Next: stop the table and add a little lower framing margin, then repeat
the same stopped bracket to fresh outputs. No special pocket angle, new caliper
reading or USB reconnect is required. No production change or M27 resumption.

Latest visibility update, 2026-09-09: the
[exposure check and second-view repeat](component-pocket-exposure.md) consistently
recover a coherent partial pocket floor at 10 ms exposure, fixed 20 ms frame
period and gain 24. All 120 short-exposure frames support raw-plane measurements;
primary medians are 11.974/11.977/11.975 mm versus the owner's 12.10 mm, with
meaningful spatial-support dependence. No tracking or fitted scaling is used.
The reconnect shifted the scene; the second block is analyzed with explicitly
reselected floor bounds, not falsely claimed as an unchanged-pose repeat.

Correction: the earlier `base` return-control patch is on the turntable. Its
loss does not establish a printed-base coverage penalty. Two actual printed-
base patches return 100% at both exposures, with much better plane residuals
at 10 ms. The mistaken external-height interpretation and all its results are
preserved and excluded from component dimensions. The corrected quiet adapter
completes one clean 266-frame hardware run, original settings restored, maximum
CPU/NPU 60.2/59.5 C; the first wrapper's USB failure remains recorded. No global
preset, production change, merged exposure cloud or M27 resumption. Next: whole-
object depth framing with the table stopped, then a short range-specific bracket
before a full-turn test against existing X/Y calipers. No new caliper is needed
first; the current close view crops the object and cannot qualify its full size.

Latest localization update, 2026-09-09: the
[offline size-bias controls](component-size-bias-results.md) rule out the tested
point-density, depth-clipping and rigid-coordinate handling choices as the main
cause: their largest width effect is under 0.008 mm. A separate raw-plane
centre/axis calculation changes widths materially but is itself model-sensitive;
neither a corrected reconstruction nor a hardware fault is established. Original
clouds/calibration remain unchanged. Next: stop the table and supply a supported,
steep downward view into the pocket for a bounded single-frame depth/coverage
check. Its existing 12.10 mm measurement will not by itself qualify X/Y scale.
No new acquisition, production change or resumption of M27 occurred in this phase.

Previous execution update, 2026-09-09: the
[controlled fresh repeat](component-repeat-results.md) now completes without
fixed-axis fallback: 809 raw frames, 405 predeclared comparison frames, no drops,
and scanner temperatures no higher than 61.6/60.8 C. Its X/Y dimensions differ
from the saved baseline by about 0.10 mm but remain 0.660/0.572 mm short against
the owner's new mid-height 91.77/81.74 mm calipers. Observed-surface RMS is
0.443 mm; the pocket floor is entirely unobserved. The constrained path is
promising, not dimensionally qualified; isolate the remaining size bias and
agree the clearance tolerance before accepting a chassis component. No fitted
scale correction, experimental mesh or production change was made.

Earlier execution update, 2026-09-09: the operator authorized the bounded saved-capture
check. Its [results](component-baseline-results.md) recover the historical
approximately 0.50 mm shape score while exposing 0.8-0.9 mm undersized base
dimensions. The unmodified evaluator timed out; a separately recorded,
equivalence-checked reduced-SVD evaluation completed on its unchanged cloud and
poses. A FreeCAD overlay is saved. At that checkpoint no fresh physical scan or
production change had run; the fresh-repeat result above supersedes that status. The proposal below
is retained as the pre-execution decision history, not the latest run status.

Operator clarification, 2026-09-09: set mapper research aside. The immediate
need is to scan physical components so they can be positioned on the chassis;
manual measurement is possible, but scanning should reduce the work, not add
an open-ended research burden. This is a proposed workflow, not an executed
scan or a resumption of M27. Preserve all earlier research and failure evidence.

Subsequent operator clarification: Revo Scan does not support Linux, and the
Android build was already used as a reference roughly one or two weeks earlier
without resolving reliability. Treat Linux capture as a settled requirement,
not an unanswered platform choice. The earlier official-software-first proposal
is superseded; neither another OS nor another vendor-binary review is the next
required step.

## Why the current workflow has become difficult

This repository implements a complete Linux-native capture, registration,
reconstruction and GUI path without the vendor SDK at runtime. It is not just
a Linux front end for the proprietary Revo Scan tracker. The current tracking
failure therefore entangles raw measurements, target/scene conditions and our
own estimation pipeline. False motion also appears in stationary saved inputs;
operator hand steadiness alone cannot account for it.

No quality-qualified general handheld fix exists. Higher frame rate and complete
acceptance have previously hidden geometric errors. The approximately 0.50 mm
CAD result in `turntable-registration.md` is a result on a particular earlier
capture, not general validation of arbitrary components or later captures of
the target. The existing turntable estimator infers its axis/rate from the data;
it is not an independently measured indexed fixture.

## Physical reference and CAD tools already available

The operator reminded us that the printed target's major dimensions had already
been measured and compared with nominal CAD. The recovered owner record is
[`physical-print-20260831.json`](../reference/targets/reconstruction_v1/prints/physical-print-20260831.json),
for `reconstruction-target-v1`:

| Dimension | Nominal CAD (mm) | Measured print (mm) | Print minus nominal (mm) |
| --- | ---: | ---: | ---: |
| Overall X | 92.00 | 91.90 | -0.10 |
| Overall Y | 82.00 | 81.85 | -0.15 |
| Overall Z | 56.00 | 55.85 | -0.15 |
| Pocket depth | 12.00 | 12.10 | +0.10 |

The predeclared reference scale factors are X 0.9989130434782608,
Y 0.9981707317073172 and Z 0.9973214285714286. `tools/target_gate.py` already
applies them in `load_reference()`; applying the print correction is not a new
untested idea or an omitted step in the historical CAD comparison. Pocket depth
is a separate measurement, not a reason to warp local CAD to force agreement.
Keep these physical measurements fixed during evaluation; do not fit scan scale
to hide a discrepancy. Bind captures to the correct physical target before
using this record; do not apply its correction to unrelated objects.

One reading per dimension, unrecorded caliper resolution and incomplete print
metadata still limit a formal accuracy claim. They do not make this reference
unusable for a practical comparison or justify asking the owner to repeat all
the preparation before diagnosing millimetre/centimetre errors. The measured
overall size offsets are small; they do not explain those much larger tracking
failures. Conversely, four readings do not prove that every local surface of
the print equals the axis-scaled CAD.

Read-only application inspection confirmed FreeCAD 1.1.3, installed as the
system Flatpak `org.freecad.FreeCAD`. It can be considered for visual inspection
and CAD import checks; no application was launched or CAD document modified in
this discussion. Numerical comparison can also use the existing reference PLY
and evaluation code without requiring GUI work.

One reporting caveat found while checking this path: the old gate report and
documentation say rigid registration absorbs any global scale error. That
sentence is mathematically incorrect. The inspected `place()` path applies
rotation and translation after the fixed print correction; rigid motion cannot
change pairwise distances. Partial coverage, nearest-surface matching and
observation thresholds can still make its aggregate score a poor dimensional
test, so explicitly compare measurable dimensions as well. Record this wording
correction here without modifying frozen executable sources or old evidence.

## Proposed first milestone

Produce one useful, dimension-checked component reference in the owner's CAD
workspace, then reproduce it without unexplained resets or special recovery.
The needed deliverable is an adequately complete external shape for packaging,
not necessarily a watertight, textured or parametric reverse-engineered model.
Agree the layout/clearance tolerance before accepting the result. Verify units,
overall size and several independently measured dimensions after CAD import.
Use measured mounting datums, hole spacing and critical interfaces where needed;
do not treat smoothing or hole filling as measurements of hidden geometry.

## What the earlier Android reference does and does not establish

The historical `.coordination/planner/delivery-roadmap.md` entry dated
2026-08-29 names `RevoScan_V6.0.3.5_202608071750_default_release.apk` as a
read-only behavioral/architecture reference. It identifies stripped ARM64
tracking, coarse/fine registration and fusion libraries, including
`libRPAFusionTracker`, `libRPARegisterCoarse`, `libRPARegisterFine` and
`libRPAPointCloudFuse`, rather than portable application source. The historical
workflow instructions there are not active task controls.

That entry records the intended reference approach; it is not an independently
audited account of every experiment performed. The operator reports that the
effort did not solve the problem. Studying binary interfaces and architecture
does not establish an equivalent Linux implementation or recover every
preprocessing, calibration, rejection and recovery decision. It also does not
establish whether the official Android app was tested on this exact scene and
unit. Do not turn the absence of that comparison into another prerequisite for
the owner's Linux workflow, or treat the failed reference effort as proof of a
hardware defect. No APK was opened or executed during this discussion.

## Proposed bounded Linux route

### Immediate next checkpoint: reproduce the documented positive capture

Read-only inspection located all 190 raw frames in
`/home/ben/scans/sweep/20260904-020845`, the capture explicitly identified as
the measured CAD target in `docs/target-gate.md`. That history reports the
fixed-axis path at 0.50 mm RMS, p95 0.89 mm, 95% observed and worst face
orthogonality deviation 0.75 degrees. These are historical development metrics,
not newly reproduced results or a dimensional-accuracy guarantee. The existing
`reconstruction-0001/reconstruction.json` describes a frame-chain reconstruction
and internal readiness checks; it must not be substituted for the documented
fixed-axis CAD evaluation just because it belongs to the same capture.

The next proposed execution is one current-code, fixed-setting replay using the
existing read-only target evaluator: explicit 60-300 mm depth band, placement
seed 5, turntable path enabled, no old cloud cache. Preserve raw inputs, file
times, existing reconstruction outputs and frozen M27 sources. Record the actual
registration path, input/source identity, runtime, geometry, missing surfaces
and any difference from the historical rounded figures. No parameter matrix,
firmware work, device acquisition or experimental mesh is needed for this check.
This replay has NOT been started during the discussion.

If the historical result reproduces, the next acceptance work is explicit
dimension/coverage checking and a CAD overlay of the output, followed by a
fresh controlled repeat with the same measured target before moving to a real
component. Keep print correction fixed; report an unobservable dimension as
unobservable rather than estimating it from the CAD or a filled surface. If
the baseline does not reproduce, localize that discrepancy before expanding
the task. A nominally good CAD nearest-point score alone is not acceptance.

This replaces the suggestion to start a new generic single-view-versus-fusion
investigation. The target-gate history already tried that comparison and later
withdrew conclusions contaminated by incorrect CAD placement. Its September 3-4
evidence implicated registration on those captures, without establishing perfect
raw depth. Read the later corrections before treating earlier headings as current
findings. The newer tracking corpus is not automatically the same physical
target; its missing identity association is separately recorded in
`20260908-reliability-mechanisms/cad-reference-readiness.json`.

### Subsequent decisions

1. Start with the already measured printed target and its existing raw captures;
   recover the prior CAD comparisons before deciding what diagnostic is still
   missing. Compare suitable individual views and reconstructed geometry against
   the fixed, print-adjusted reference and independently measured dimensions.
   Label dimensions that a view does not expose as unobservable. A stable-looking
   view alone proves neither scale nor accuracy; if individual-view checks fail,
   isolate acquisition/calibration before adding multi-view alignment. New
   measurements are not a blanket prerequisite. The later representative chassis
   component still needs its own relevant dimensions and layout tolerance.
2. Assess the existing offline fixed-axis reconstruction, starting with its
   historical positive fixture as a regression reference and a suitable complete
   turn of the representative component. This path already exists; do not sell
   it as a new algorithm. Explicitly record whether it actually used the axis
   model or fell back to the general tracker. The recent live-tracker replay
   campaign and offline axis reconstruction are different paths. Agreement after
   one model-enforced revolution is not an independent shape-accuracy test.
3. If independently checked dimensions and repeatability meet the agreed need,
   make that narrow path the component workflow: fixed scanner, controlled slow
   turn, separate captures for different tilts, offline processing and checked
   CAD import. Verify sufficient coverage for the intended clearances. Do not
   make a general freehand tracker or 10 fps a prerequisite for this deliverable.
4. If raw geometry is adequate but the fitted rotation/axis is not, consider
   stopped, indexed views with independently measured fixture poses. This is
   materially different from inferring rotation from the same clouds, but is
   not implemented or validated. It needs a suitable fixture, axis/angle
   calibration, uncertainty checks and likely operator action. A stopped table
   or a nominal motor speed alone does not provide a known pose. Known poses
   cannot repair inaccurate raw depth.

The acceptance result is a useful exported reference with independently checked
dimensions, followed by a repeat scan that meets the same tolerance. It is not
just successful execution, a visually smooth surface or a closed revolution.
Retain missing surfaces as unknown; no experimental mesh work is part of this
proposal. No new scan or reconstruction has been executed for this revision.

Official references retained from the superseded platform proposal:
[original MINI support](https://www.revopoint3d.com/pages/support-mini),
[current compatibility/download table](https://www.revopoint3d.com/pages/support-download),
[object and marker guidance](https://revopedia.revopoint3d.com/en/scanningtips/objecthandling),
[Revo Scan 5 model editing](https://revopedia.revopoint3d.com/en/Revoscan/Revoscan5usermanual/Modelediting).
These are background manufacturer descriptions, not a Linux execution route or
measurements of this unit. Web search exposes the editing description; the direct
Revopedia page yields only its minimal page shell in the text reader.

## Decision and stopping points

Start with a bounded qualification of the existing constrained path, not another
broad tracker matrix. If it fails, report whether the failure is already visible
in independently checked single views or appears when views are combined. If
the necessary physical reference or representative capture is absent, identify
that concrete operator task and pause. Do not use parameter sweeps as a substitute
for missing measurement evidence. Existing frame-derived turntable fits must not
be relabelled as externally known poses. No hardware diagnosis or reliable
general freehand solution is established by this proposal.

Do not restart FPS optimization, firmware investigations or experimental mesh
development before the component workflow has justified them. No vendor software,
driver, scanner setting, firmware, calibration or mapper file changed during this
discussion. The suspended batch and all source/output identities remain governed
by `reliability-discussion-checkpoint.md` until the operator chooses how to proceed.
