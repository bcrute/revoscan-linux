# Stationary physical-reference readiness — 2026-09-10

Current result: the [more downward staircase view](component-staircase-downward-results.md)
has broader image support and about 9.931 mm primary spacing versus provisional
9.90 mm. The same-pixel control confirms 10 ms improves local surface shape,
but physical-reference and patch/direction uncertainty still prevent an accuracy
decision. Keep this scanner angle; request repeated underside-to-top heights
at identified front/centre/back tread positions and caliper information. No new
print or further scanner reposition is the next dependency. Earlier actions follow.

Latest execution: the [first new staircase view](component-staircase-results.md)
has both tread tops and completes a clean 20/10/20 ms bracket. All four fixed
patch combinations support fits in 120 frames, but a primary 10.243 mm estimate
versus provisional 9.90 mm is accompanied by nonparallel fitted planes and
substantial evaluation-location/direction dependence. Exposure does not resolve
it. Next is a more downward supported/stopped view with broader tread interiors,
not a new print. Original settings restored; maximum CPU/NPU 59.8/59.1 C.
The following measurement/readiness entries retain their earlier next actions.

## Later owner measurements: reuse the existing staircase first

The owner would have to print a new reference and questioned its necessity.
A larger step is a cleaner diagnostic option, not an established requirement
for continuing. Hold off on a new print while checking the existing staircase.

The owner then supplied **16.08 and 25.98 mm**, explicitly from the object's
bottom to the tops of the two highest staircase treads. These are recorded
verbatim in the [supplemental physical-print record](../reference/targets/reconstruction_v1/prints/physical-print-20260831-steps-20260910.json).
They map to nominal total heights 16 and 26 mm (`step_10mm_top` and
`step_20mm_top`); the CAD surface names encode heights above the nominal base.
The 16.08 mm reading is **not** the height of the separate pocket plateau above
the printed base. Do not subtract a nominal 6 mm base thickness to invent
physical above-base measurements. The original print record and CAD adjustment
remain unchanged.

The arithmetic difference is **9.90 mm**, a provisional reference for a
comparison between two visible tread tops. That avoids needing to observe the
underside in the scan, but assumes a common physical underside datum and
correspondence between the caliper contacts and scanned patches. Exact contact
locations, repeated readings, instrument resolution/accuracy, underside flatness
and tread parallelism are not yet established. No new accuracy verdict follows;
0.3% of the small difference is only 0.0297 mm, not a demonstrated uncertainty.

A bounded visual recheck of the existing stopped full-view image, both pocket
views and the September 8 centered view did not establish a clearly observable
pair of these tread interiors. The first three views emphasize the other side
of the ramp; the older centered view presents the staircase from its high end
without a clearly exposed lower tread interior. This is visual readiness only,
not a numerical support test or an exhaustive search of historical frames.
Reviewed images: `stopped-retry-1tOcGP/stage-1-frame-00.png` in the sealed
full-view root, `exposure-a/stage-3-exposure-10000.png` and
`exposure-repeat/stage-1-exposure-10000.png` in the sealed pocket root, and
`reviewed-context/images/05--0000-depth.png` in the sealed readiness root below.

Next setup: supported scanner, stopped turntable, staircase facing the camera
with both measured horizontal tread tops exposed. The underside and pocket
floor need not be visible for the tread-to-tread comparison. Live framing has
not been checked and no new capture, numerical fit, calibration or production
change occurred while recording these readings. Use the next view to establish
surface support before asking for more physical-reference detail or claiming
the existing object is sufficient for calibration. The following sections
retain the earlier sealed checkpoint, including its then-current next action.

## Outcome and required next action

The reviewed saved stationary views do **not yet supply a qualified independent
lateral reference** for fitting and checking a transferable calibration
correction. The missing pieces are physical datum correspondence and observed
support, not another replay of the same turntable-derived widths.

A promising-looking alternative was checked rather than assumed: the outer
rectangle of the raised pocket block is visually recognizable in two stopped
views, but two of its four perimeter guides border predominantly missing depth.
The fixed support test refuses **99.3–100%** of transects on those two edges in
each stage. The interior plane is well supported in all 120 frames. Thus a clean
interior and a plausible outline do not establish a metrologically supported
perimeter. No physical rectangle width or calibration coefficient was fitted.

Do **not** request new caliper dimensions of that outline as though those alone
would close the problem. The next physical dependency is a larger, measured
parallel-plane step with both levels visible. The owner has been asked whether
a rigid, flat, matte block/box approximately 40–80 mm thick is available to
measure and place on a flat matte supporting surface. Its shape, readings and
setup are not yet supplied. No new capture or scanner operation occurred here.

No [component acceptance gate](component-acceptance.md) passes. This result does
not identify defective hardware/firmware or promote a scale/processing change.
It identifies why the currently reviewed references cannot settle that choice.

Evidence root: `/home/ben/scans/diagnostics/20260910-stationary-datum-DqRI12/`.
Sealed at 2026-09-10 15:49:14 UTC: **164 files / 47,944,165 bytes**, excluding
the manifest itself; manifest SHA-256
`dea099357f786fde4ce0130d2e2655bcb4d5686e6b732cf9088ca5fc4190c5cf`.
Exact file-set/size/hash verification passes. Do not modify the sealed root.
Archived documents precede this seal metadata. Closeout verifies both preceding
can roots, all 78 live snapshot sources and all 449 frozen live/saved context
files unchanged; PID 2009261 remains stopped.

See `PROTOCOL.md`, `support-report.json`, `support-audit.json`, all 120 per-frame
records, `perimeter-support.png`, reviewed context and `NEXT-REFERENCE-CHECK.md`.

## Physical-reference crosswalk

The physical print record and subsequent mid-height clarification are preserved
separately. The latter has one owner-reported value per axis, without instrument
resolution, repeated readings or a calibrated uncertainty. Extra CAD decimals
and a tight same-recording frame spread do not supply that missing information.

| Reference | What the saved stationary evidence provides | Remaining limitation |
| --- | --- | --- |
| Base X/Y 91.77/81.74 mm at sidewall mid-height | Visible adjacent walls and partial perimeter; multi-view records contain the other walls | Opposing-plane separation currently needs estimated relative poses or fixture/shape assumptions. A different-height top outline is not this datum. |
| Earlier overall X/Y 91.90/81.85 mm | A distinct physical reading, retained unchanged | It must not replace the clarified mid-height reference merely to match a different edge. |
| Overall Z 55.85 mm | Ramp/highest-edge depth; underside rests on support | Hidden bottom is not directly observed; transfer from a support plane requires justified contact/flatness, and a peak edge is not a second broad plane. |
| Pocket depth 12.10 mm | Already directly measured in 120 short-exposure frames across two views | Small, partly observed, nonparallel/patch-dependent separation; about 0.08 mm plateau-patch dependence versus a 0.0363 mm working limit. Not a lateral-scale check. |
| Pocket-block outer top spans | Nominal rectangle identity and a well-returning interior in two views | No physical calipers for these spans; the new perimeter-support check also fails two edges. |
| Printed-base to plateau step | Previously fitted plane separations, nominally 16 mm | No corresponding physical height reading; support choice changed earlier estimates by about 0.23 mm. Nominal CAD is not calibrated truth. |

For a convex rectangular base, an east exterior plane is front-facing only
when the camera is outside its east side, while the opposite west plane is
front-facing only when the camera is outside the west side. Those conditions
cannot both hold at one pinhole viewpoint. The same applies north/south. This
is a limitation of the opposing-plane measurement, **not a proof that every
single-view edge/silhouette method is impossible**. Such an alternative needs
its own edge correspondence, actual physical section and bias/uncertainty check.

Visual context reviewed includes the September 8 centred stopped view,
September 9 setup RGB, the two pocket views and the latest genuinely stopped
full-view image. This is a bounded review of the relevant saved target context,
not a claim that every frame in the entire historical archive was inspected.
The quantitative test below uses only the explicitly named pocket blocks.

## Fixed raw-data perimeter-support test

Inputs: all 40 retained frames from each of:

- `exposure-a/3-20000`: first physical view, 10 ms exposure.
- `exposure-repeat/1-20000`: reconnect view, first 10 ms stage.
- `exposure-repeat/3-20000`: same reconnect view, second 10 ms stage.

These are **120 old frames, two physical views, and three exposure stages**,
not three independent acquisitions or a controlled range sweep. The first
block's historical USB-close failure remains recorded in
[the pocket experiment](component-pocket-exposure.md); it is not relabeled a
clean lifecycle by this offline analysis. The two poses do not identify exactly
the same physical patch at one fixed image ROI.

The plateau plane uses the original `[146,151,196,201]` interior and unchanged
saved plane helper. Intrinsics and raw scale are the exact recovered factory
values. Zero/65535 codes are treated as missing/reserved as in that saved pocket
analysis; zero 65535 codes are encountered in this test. No temporal median,
inter-frame alignment, imposed parallelism, CAD placement or scale is used.

After visual selection and before numeric execution, four approximate guide
corners per view were fixed in `PROTOCOL.md`. They identify search strips;
they are not fitted corners, a rectangle constraint or physical dimensions.
Each edge has 25 transects over its central 60%, avoiding corners. Each transect
has 33 ordered samples at offsets -16 through +16 image pixels along the guide's
inward normal. All rounded sample coordinates and any duplicates are retained.

Valid returns within 0.30 mm of the fitted top plane are tagged top support;
valid returns at least 1 mm away are tagged other-surface support. Intermediate
returns and no-return samples remain separate. Require at least four top
samples at inward offsets +9..+16 and four other-surface samples at outward
offsets -16..-9. Otherwise refuse. On supported transects retain the last
outside other-surface observation and first run of three top samples. No
threshold or guide was adjusted after seeing these outcomes.

The between-sample span is also projected onto the fitted plane. It is a
**sampling/visibility transition span**, not an error bar, calibrated edge
interval or proof of where a physical boundary lies. Surface shadows, mixed
pixels, interpolation and print shape remain possible influences. A subpixel
line fit could produce a small residual without resolving those causes.

## Results

The interior plane passes support in all 120 frames. Its all-selected-point RMS
medians are 0.03626 mm in the first view and 0.04765/0.04647 mm in the reconnect
view stages. These are local surface residuals, not width uncertainty.

Each edge has 1,000 transects per stage (25 per frame):

| Stage | E0 supported | E1 supported | E2 supported | E3 supported |
| --- | ---: | ---: | ---: | ---: |
| First view, 10 ms | 1000 | 1000 | 1 | 1 |
| Reconnect view, first 10 ms | 1000 | 1000 | 1 | 5 |
| Reconnect view, second 10 ms | 1000 | 1000 | 0 | 7 |

Every transect has sufficient inward top support. E2/E3 refusals are specifically
insufficient **outward observed-surface support**, with median eight missing
samples out of the eight outer samples. They are not caused by a failed plane
fit or a guide wandering into the pocket. Sparse accidental supported transects
remain recorded rather than being used to declare an entire edge measured.

Median projected transition spans for E0/E1 are 0.6316/0.5719 mm in the first
view and 0.4966/0.5697, then 0.4957/0.5706 mm in the reconnect view. Most E1
transitions include missing samples. These numbers describe observed sampling
around the guides, **not actual dimensional errors or proven uncertainty
bounds**. The protocol makes no subpixel metrology claim from them.

The selected interior's median camera-coordinate Z is 185.802 mm in the first
view and 185.681/185.675 mm in the reconnect view. This is not an independent
standoff measurement: the scene pose changed and the fixed ROI identifies a
different patch. No controlled distance/angle sweep or transferable calibration
training/hold-out split exists in these stages.

The data-derived support visualization is preserved as `perimeter-support.png`
in the evidence root. Pink lines are fixed search transects, not inferred true
edges. The dark strips remain missing depth; they are not filled for display
or for the analysis.

## Why the next reference is a step, and what it does not solve

A suitable rigid block placed flush on a flat surface presents two broad
planes: its top and the exposed support around it. Independently measuring
block thickness gives a separation without requiring a far outline, a cylinder
radius fit, multi-view tracking or nominal print dimensions.

This involves an explicit **physical datum transfer**: the hidden underside
must actually contact the same plane represented by the visible surrounding
support. No feet, gap, rocking, bowed lid or warped/compressible support may be
silently ignored. Surface flatness/parallelism, reading locations, at least
three individual readings and instrument resolution/uncertainty must be recorded.
The owner has been asked what suitable object is available before imposing a
new print or purchase. No candidate or reference value is assumed yet.

At 40–80 mm separation the 0.3% working limit is 0.12–0.24 mm, compared with
0.0363 mm on the pocket. This gives a larger physical comparison; it does not
guarantee success. A face-on plane-spacing result is primarily a depth-direction
check. Tilted views in both directions and different ranges would subsequently
be needed to test lateral/directional and range dependence. Those conditions,
the measured datum and any candidate model must be fixed before scoring, with
held-out geometry retained. A passing height alone does not qualify X/Y, prove
one global scale correction, or complete the component/CAD workflow.

The immediate blocker is therefore specific: identify, measure and position a
suitable reference with justified support contact. No new scan is needed merely
to repeat the unsupported plateau outline or the already-known can undersizing.

## Verification and preservation

All **12 deterministic diagnostic tests** pass, covering ray-plane geometry,
classification thresholds, missing-depth refusal, sample order, the existing
plane helper, and the opposing-exterior-plane visibility condition. These are
software tests, not accuracy qualifications. The one numerical run completes
in **4.1755 seconds**; no numerical retry, threshold sweep or calibration fit.

The audit verifies all 120 input hashes and frame records, **12,000 transects /
396,000 ordered sample records**, classifications, all-input plane residuals,
transition/refusal outcomes and stage summaries. All 135 execution-bound
inputs/sources/references match before canonical document updates, with reviewed
context preserved separately. The pocket bundle's exact **593-file** set over
both pocket roots and **237,584,657 bytes** verify against manifest SHA-256
`13e9db018ec45187127236fba1f480f9c39cf5618fbc2be3af5e07278777f3e0`.

A presentation-only plotting import failed because matplotlib was absent; a
Pillow availability check also found it absent. No dependency was installed.
The original plotting source/note are retained and the existing offscreen Qt
renderer successfully produces the direct raw-data plot. Numerical results
were not rerun or changed. No scanner/temperature access, production/firmware
change, experimental mesh or M27 resumption occurred.
