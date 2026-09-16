# Reliability learning index

Current pause point: [2026-09-10 scanner checkpoint](scanner-checkpoint-20260910.md),
prepared for the owner's second-opinion review. It records the pending physical
measurements, latest completed results, evidence paths, freshly checked host state
and preservation cautions. No further experiment was started for that checkpoint;
earlier “next” actions below are historical, not an outstanding task queue.

Latest bounded result, 2026-09-10: [second, more downward staircase view](component-staircase-downward-results.md).
Clean 20/10/20 ms bracket: 160 full/120 settled frames, no drops/shorts or
transport fault, original settings restored, CPU/NPU peak 58.9/58.2 C.
Primary 10 ms spacing is 9.931477 mm versus provisional 9.90 mm, closer than
the first view's 10.243219 mm; angle, range and physical patches changed, so
this is not an isolated camera-angle effect. All 480 fixed pair fits have
support. Remaining patch/direction dependence prevents calibration/accuracy
qualification. A predeclared post-primary common-pixel control (975 upper,
701 lower) retains 120 fits: lower all-point RMS 0.24268/0.02717/0.24262 mm
and free angle 2.948/0.816/3.096 degrees. Changing missing-pixel selection alone
does not explain that exposure effect; cleaner data is not absolute truth.
The same 27 existing tests pass, both raw/result audits pass, and no production,
calibration, firmware, mesh or M27 change is made. Next: location-identified
repeat calipers and instrument information; keep this scanner angle, no new
print or immediate reposition. Evidence:
`/home/ben/scans/diagnostics/20260910-staircase-downward-iRDqsk/`.
Sealed 17:35:03 UTC: 407 files / 76,784,405 bytes excluding the manifest,
SHA-256 `796a3df3b48e79c8a8416606bf6d6312b4cc8f046a24dd860a496c7bf30f67e4`.
All bound sources/inputs, the first-view sealed root and 449 frozen live/saved
source records verify unchanged; M27 remains stopped.
An early renderer invocation preceded report creation, failed without hardware
access/output, then succeeded after acquisition completion; no capture retry.
Earlier checkpoint-specific next actions follow.

Latest bounded hardware/raw-plane result, 2026-09-10: [first staircase view](component-staircase-results.md).
The existing target exposes both measured treads. A 20/10/20 ms bracket records
160 full deliveries/120 settled frames without drops/shorts or transport fault;
original settings restored through the quiet boundary, CPU/NPU peak 59.8/59.1 C.
All selected pixels return, and all 480 fixed pair fits have support. Primary
medians 10.243787/10.243219/10.242364 mm do not match provisional 9.90 mm, and
the independently fitted planes are nonparallel: primary reverse evaluations
are about 10.58 mm; shifted-patch reverse medians reach 10.84 mm. This is not
a defensible scalar calibration height or a reason to average conventions.
Exposure does not fix these particular strips. Next is a more downward
supported/stopped view with broader tread interiors, not a new print or global
scale correction. Twenty existing tests and seven new reporting tests pass;
all 120 raw inputs/480 pair results and selected samples audited. Evidence:
`/home/ben/scans/diagnostics/20260910-staircase-check-0aO0EY/`. No production,
firmware, mesh or M27 change. Earlier checkpoint-specific next actions follow.
This root is sealed at 17:16:40 UTC: 383 files / 72,876,568 bytes excluding the
manifest, SHA-256 `baefc490af7adfa4a141e476e3f2d771a4c1b84b98030cb0207605a3a71ce649`.
All bound inputs and 449 frozen live/saved source records verify unchanged.

Latest owner-measurement update, 2026-09-10: [existing staircase reference](stationary-reference-readiness.md).
Do not require a new print before checking the existing target. The two highest
staircase treads were reported as **16.08/25.98 mm from the physical underside**,
not above the base top and not the pocket plateau. A separate [measurement
record](../reference/targets/reconstruction_v1/prints/physical-print-20260831-steps-20260910.json)
preserves the verbatim message, CAD surface identity and original print record.
The difference **9.90 mm** is provisional common-datum arithmetic, not a
certified plane spacing. No nominal base subtraction, rescaling or accuracy
pass. A bounded visual recheck of four existing stationary poses did not
establish both tread interiors; request a staircase-facing supported/stopped
view, not visibility of the pocket or underside. No new hardware or numerical
test ran; measurement locations, repeats and instrument uncertainty remain open.

Earlier checkpoint-specific next-step wording follows.

Latest bounded result, 2026-09-10: [stationary physical-reference readiness](stationary-reference-readiness.md).
The relevant saved views do not yet provide a qualified independent lateral
calibration reference. Physical mid-height X/Y calipers remain distinct from
top/overall outlines, CAD and turntable-model-derived widths. The proposed
outer pocket-block perimeter was not assumed measurable from its appearance:
120/120 top-plane fits pass, but E2/E3 outward support fails 99.3–100% of each
stage's 1,000 transects per edge. E0/E1 are supported, with roughly 0.50–0.63 mm
median projected transition spans; these are sampling descriptions, not physical
edge errors or calibrated uncertainty. All 12,000 transects / 396,000 ordered
samples are audited. No dimension, subpixel edge, scale or calibration was fitted.

The two pocket poses are not a controlled range sweep. No plateau-outline
caliper request is made merely to put truth numbers onto unsupported boundaries.
The next physical dependency is an available rigid matte block/box roughly
40–80 mm thick, caliper-measured and flush on a flat matte support. This exposes
two broad planes with an explicit contact-based transfer of the hidden underside
datum. A face-on height alone is not lateral calibration; directional/range
and held-out component checks remain necessary. Owner object/setup is requested.
Twelve tests pass; numerical execution 4.1755 s; no retry/threshold search. The
matplotlib plotting import and Pillow availability check fail without affecting
analysis; existing offscreen Qt produces the data plot, no dependency install.
Evidence: `/home/ben/scans/diagnostics/20260910-stationary-datum-DqRI12/`.
Sealed at 15:49:14 UTC: 164 files / 47,944,165 bytes excluding manifest,
SHA-256 `dea099357f786fde4ce0130d2e2655bcb4d5686e6b732cf9088ca5fc4190c5cf`.
Exact set/size/hash verification passes; both preceding can roots, 78 live
snapshot sources and all 449 frozen live/saved context files verify unchanged.
No scanner/temperature access, firmware/production change, mesh or M27 resumption.

Earlier checkpoint-specific next-step wording follows.

Latest bounded offline result, 2026-09-10: [can spatial support and residual-tail
influence](can-support-results.md). The tested missing-pixel patterns alone do
not explain the usable frames' size error on ideal geometry: exact-original-pixel
controls recover 58 mm within 0.000213 mm for all 14 historically usable cases.
An audit-driven, separately declared exact-pixel addendum was needed because
percentile/crop/stride selection on ideal values can change final pixel sets.
The original procedural controls and both negative controls remain recorded.

One fixed six-MAD tail refit makes bare-can means generally smaller; the
sprayed-far mean changes only -0.003481 mm to 57.713105 mm, 0.49465% short.
All 16 real point estimates remain outside 0.3% and all preserved-style
comparisons are inconclusive. Bare-120 spread tightens from 0.043920 to about
0.001185 mm while its mean error worsens to 0.541573 mm: cleaner/tighter fitting
is not external dimensional accuracy. Spatially structured real residuals remain;
this does not identify firmware/calibration/material as the cause. Seventeen
tests pass; 97.6519 s primary execution and 32.9783 s exact-support addendum,
no process timeout/failure, one explicit exact-support refusal for 692 missing
ideal intersections in the rejected sprayed-125 case. No real-data retry,
filter promotion, scanner access or production change. Next: establish whether
saved stationary printed-target data provide an independently observed physical
lateral-reference comparison across view/range; name any missing datum before
asking for new capture. Full evidence is in
`/home/ben/scans/diagnostics/20260910-can-support-NKJ6NM/`.
Sealed at 15:19:59 UTC: 236 files / 75,462,705 bytes excluding manifest,
SHA-256 `17bd8796d51c6ee65516200f4d093f2def3061dfba9fbb81f8b0b56b9f197d55`.
Exact set/size/hash verification passes; 78 live snapshot source files and all
449 frozen live/saved context files are unchanged; M27 remains stopped.

Earlier checkpoints and their original next-step wording follow.

Latest bounded offline result, 2026-09-10: [original can replay](can-replay-results.md).
The old scanner-specific task record recovers the exact filtering/stride/crop
commands and their outputs; retained Git history binds all 16 raw files,
including the separately rejected sprayed-125 frame. Thirty-two CPU-only
historical/current replays finish in 41.33 seconds, all 16 original per-frame
diameters match their recorded rounding, and geometry/selection parity is exact.
All fits converge, all historical/new-threshold validation verdicts remain
inconclusive, and no point-estimate diameter meets the below-0.3% criterion.
Thirteen distinct wrapper/audit tests pass. No production/device changes.

The missing method is no longer a blocker. Important denominator corrections:
the 7-frame near-direct mean has only 3 retained direct frames; both 4-frame far
means have only 3 retained frames; the rejected sprayed-125 mean and its quoted
fit uncertainty came from different aggregation levels. A clean synthetic
cylinder is not a control for real uneven/noisy support, so the old ideal-case
test did not eliminate all host-side fitting bias. The next selected question
is saved-data spatial-support/outlier sensitivity, not another generic can
capture or an inference that firmware is faulty. The full result, command
extracts, both source trees and raw bytes are preserved in the new evidence root.
That root is sealed: 306 files / 15,248,270 bytes, manifest SHA-256
`d3d97cda195bdd9c73fc938047fcc7ad5fbfd1302698356884559ef5a216b553`.
An initial final-check list-order mismatch is documented in the result; the
independent read-only exact-set/size/hash verification passes. All 449 frozen
live/saved context files are unchanged and M27 remains stopped.

Current acceptance decision, 2026-09-10: the owner accepted the
[fixed component criteria](component-acceptance.md). The current dimensional
results do not meet the below-0.3% requirement. Repeatability, transfer to a
different component, observed interface coverage and the usable Linux workflow
remain distinct gates. Next is a bounded offline recovery/replay of the August
24 can procedure, using its preserved inputs and later corrections before any
new stationary experiment. No scanner or production change accompanies this
checklist update. Earlier dated entries below retain their original status.

Latest turn-test checkpoint: [full-turn exposure comparison](component-turn-exposure.md).
The clean 2,470-delivery bracket preserves 2,430 unique settled raw frames,
actual callback times, occasional explicitly labeled live framing mirrors and
all sources. No drops/shorts/transport fault; original settings restored; peak
CPU/NPU 63.8/63.0 C with positive late slopes, not thermal equilibrium. Upper
ramp margin remains limited. The four-round 10 ms model is about 3 mm undersized
and 1.154 mm CAD RMS, versus baseline RMS 0.453/0.469 mm and X/Y shortfalls
0.54–0.83 mm. Better nearby-frame/loop agreement conceals worse wide-angle shape.

A concrete estimator-limit failure is isolated: the short-exposure fit hits
MAX_ROUNDS=4 with a 6.278 mm centre step despite a 0.5 mm stopping criterion;
returning a model is not convergence. A predeclared secondary continuation
reproduces every original point/pose exactly, then unchanged rounds 5/6 move
1.107/0.193 mm and stop. Evaluation improves CAD RMS to 0.482 mm and
orthogonality to 0.762 degrees but still leaves X/Y 1.100/0.997 mm short.
One predeclared combined control uses the equal-weight mean of the two 20 ms
fixture lines with ONLY 10 ms points/rate: X/Y 91.02004/81.10653 mm, within
0.003 mm of the baseline mean, still 0.74996/0.63347 mm short. CAD RMS/p95 are
0.44503/0.64547 mm, 24/24 judged surfaces, pocket floor still unobserved. This
localizes the candidate's extra undersizing to fixture estimation here without
solving or uniquely diagnosing the common bias. No production/default promotion.
Actual timestamps are supplied but all three stages trigger the unchanged
near-uniform-to-index snap; do not mistake preserved host times for unsnapped
pose angles. Earlier sources/seals remain untouched and M27 remains suspended.
All scorer exits are zero for completed evaluation even when JSON quality
verdicts fail; completion, convergence and geometry qualification are separate.
The owner has been asked which dimensional tolerance the chassis task needs.
The long bracket and both bounded offline controls are sealed: 4,742 files /
2,503,260,117 bytes, manifest SHA-256
`8446033ecd1d95c763302592400a5e865aff59e543811ca6fcd189991e0e4033`.
Exact file lists/hashes, five earlier seals, all capture/analysis inputs and
449 frozen live/context files verify. The GUI/capture process is closed;
USB identity is stable at device 34, 480 Mb/s, and M27 remains stopped.

Latest full-view checkpoint: the [stopped whole-view retry](component-fullview-exposure.md)
now verifies a stationary scene and an exposure-linked local improvement. Four
preselected usable planar patches retain their points at 10 ms and reduce full-
input plane residuals by 1.6–4.6 times, with reversal at the final 20 ms bracket.
The fifth patch returns all points but refuses every plane fit as not convincingly
planar. Matched support and all-input residuals prevent hiding losses by robust
trimming. Same-pixel depth shifts remain reported; smoother is not calibrated
absolute accuracy. The whole-image map contains 3,805 persistently lost baseline
pixels, including object-perimeter/ramp-edge areas as well as the turntable.
Do not silently classify all those losses as irrelevant background.

This fresh retry captures 160 full deliveries/120 settled frames, zero drops or
shorts, actual 4.524 fps, peak CPU/NPU 60.2/59.5 C including final readback,
original settings restored and no transport fault. The parent rotating attempt
is unchanged and still refused. First-versus-every-later frame p90 depth changes
remain at most 0.115/0.090/0.115 mm within the three stopped stages. The lower
corner still reaches the frame edge: next physical need is a little more lower
margin and starting the slow table, with fixed supported scanner and no special
pocket angle. A separate bounded full-turn 20/10/20 ms protocol/driver is prepared
but has not run. Its 18 synthetic cases plus six quality-reporting cases pass;
the first run's four test filename mismatches are retained separately. No new
caliper, firmware, calibration, default, production change, mesh or M27 resumption.

The two full-view attempts are sealed together: 485 files / 141,995,921 bytes,
manifest SHA-256 `f5948fd3345e99938c20aeca1d5bc315e360285443f2288b83f482cfd1060616`.
Exact file lists/hashes, both captures' source/snapshot records, quality inputs,
four earlier seals and all 449 frozen live/context files verify unchanged.
The separate turn-test root has 91 bound acquisition sources and no hardware
result. Read-only closeout finds the MINI on USB device 32 at 480 Mb/s and the
GUI closed; no extra control reads were issued for this checkpoint.

Previous full-view checkpoint: the [whole-object exposure attempt](component-fullview-exposure.md)
acquired 160 full deliveries/120 settled raw frames with no drops/shorts, fixed
20 ms frame period, 20/10/20 ms exposures and gain 24. Maximum CPU/NPU was
61.6/60.8 C; guarded final baseline restoration and USB identity succeeded.
Raw first/last views show continuous target rotation, contradicting the intended
stationary condition. The comparison is refused rather than attributing motion
to exposure/noise. A lower base corner reaches the depth-field bottom while
turning. Stop the table and add lower margin before a fresh stopped repeat.
All data, timestamps and source snapshots are preserved; no per-region quality,
dimension, calibration, mesh, default or production change is accepted. M27 is
still suspended. The earlier close-view pocket results remain distinct.

Latest current-view checkpoint: the [pocket exposure check and reconnect block](component-pocket-exposure.md)
recover a coherent partial floor in all 120 short-exposure frames across two
views. Primary depth medians 11.974/11.977/11.975 mm remain below the owner's
12.10 mm; alternate plateau supports move estimates by up to about 0.08 mm in
the second view. Exposure is 10 ms with frame period fixed at 20 ms and gain 24.
Long exposure repeatedly removes useful floor/top returns. The reconnect changed
framing, detected before measuring; second-view ROI selection is explicitly
recorded rather than treated as an unchanged-pose repeat.

Important semantic correction: the old `base` control is on the turntable, not
the printed base. Its 97-to-69% first-view loss is not a printed-base penalty.
The corrected second-view printed-base patches retain 100% of their points at
both exposures; primary plane RMS improves from about 0.11 to 0.025 mm, while
plateau RMS improves from about 0.52 to 0.039 mm. The first attempted external
height mistakenly used the turntable and is excluded; all original labels,
fits/refusals and the correction are preserved. Corrected base-to-plateau
height remains support-sensitive and has no independent caliper reading.

The first wrapper's post-stop USB failure is retained. The guarded quiet adapter
now completes one hardware block: 266 full deliveries, zero drops/shorts, 200
settled frames, maximum CPU/NPU 60.2/59.5 C, fresh final 20/20 ms/gain 24 readback
and stable post-close identity. Automatic re-enumeration during the quiet boundary
is recorded, not hidden. The existing 46 plus new 13 software tests still pass;
they are not hardware-trial counts. No production change, scaling, mesh, global
preset or M27 resumption. Next physical need: whole-object depth framing with
table stopped, then a short bracket and conditional full-turn comparison against
existing X/Y calipers. No new caliper reading is needed first.

Pocket/current-view evidence is sealed jointly: 593 files / 237,584,657 bytes,
manifest SHA-256 `13e9db018ec45187127236fba1f480f9c39cf5618fbc2be3af5e07278777f3e0`.
Exact file lists/hashes, the earlier three seals and all 449 frozen live/context
files verify unchanged. A fresh full-view stopped-check root is prepared with
seven passing synthetic preflight tests, but no hardware run there yet.

Latest localization checkpoint: the [component size-bias analysis](component-size-bias-results.md)
reproduces every saved prepared/CAD point exactly on both accepted captures.
All-raw density/clip controls change widths by less than 0.008 mm, compared with
0.660/0.572 mm fresh shortfalls. Saved Pl/scale constants still exactly match
software conventions, without establishing optical calibration. An independent
raw-plane centre solve changes widths materially, but three axis controls expose
remaining model sensitivity; no correction is promoted or hardware diagnosis
made. All methods, refusals and the first runner's iterator-conversion failure
are preserved. Next physical dependency is a stopped, supported downward view
of the pocket floor and plateau for one local raw-depth/coverage check, explicitly
not a complete lateral-scale test. No device access occurred; M27 remains paused.
The final 22-test diagnostic suite passes. All 449 frozen live/context files,
94 fresh capture sources/snapshots and both earlier sealed scan bundles verify
unchanged. The new size-bias bundle is sealed: 215 files / 83,420,374 bytes,
manifest SHA-256 `b19d2dcde70c095f0946f068c8426469d13bfaea24255c7f9e1b31cf1e619510`.

Previous practical checkpoint: the [controlled fresh repeat](component-repeat-results.md)
completed with 809 full frames, zero drops/shorts, 4.5166 fps and internal
CPU/NPU maxima 61.6/60.8 C. The predeclared 405-frame selection reconstructs
with the fixed-axis model without fallback. X/Y 91.10970/81.16825 mm differ
from the old scan by only +0.09401/+0.10114 mm. The owner then supplied
mid-height X/Y calipers 91.77/81.74 mm; these leave -0.66030/-0.57175 mm
(-0.7195/-0.6995%) shortfalls. Both old and new calipers remain recorded;
all eight updated selection checks reproduce exactly the original fitted faces.
No CAD scaling or geometry was changed. Surface RMS/p95 are 0.44302/0.68885 mm,
95.07% observed and 24/24 judged surfaces under the old RMS bar, but pocket
floor coverage is zero and orthogonality worsens to 1.344 degrees. This is
neither whole-target nor handheld qualification. FreeCAD points/overlay and a
direct XY projection are saved; an incomplete FreeCAD top screenshot is marked
unusable. All 94 source/snapshot files, 122 sealed baseline manifest files and
195 old capture files/times verify unchanged. M27 remains paused.

Previous setup checkpoint: the [fresh-repeat readiness check](component-repeat-readiness.md)
completed at 18:03 UTC with baseline settings unchanged, 32 depth-view frames
plus 24 depth/nine colour frames, and no depth drops/short frames. Internal
CPU/NPU temperatures ended at 58.5/57.7 degrees C. The visible target features
are consistent with the measured reference, but the scene is stationary and
the colour view clips the object's left side. Recenter the whole target and
start the turntable slowly with the camera fixed before the full repeat. No
full repeat or new dimensional comparison has run; M27 remains paused. The
record also distinguishes buffered diagnostic file-write times from actual
capture-delivery times, which must be retained for fixed-axis reconstruction.

Earlier saved-capture checkpoint: the
[measured-target baseline](component-baseline-results.md) recovers the historical
0.50 mm CAD-surface score on the September 4 capture, but direct opposing-base-face
checks read X/Y 91.01569/81.06712 mm versus calipers 91.90/81.85 mm. Both are about
0.96% short. Pocket wall still fails the old surface threshold; only 10.91% of
the pocket floor is observed. No dimensional qualification or scale correction
is claimed. The unchanged evaluator hit its 900-second limit after registration;
an equivalence-checked process-local reduced-SVD evaluation completed on the same
cloud/poses. All evidence, a FreeCAD document and scan-only/overlay views are
recorded in the new result directory. Six dimension-helper tests pass; production
sources and capture bytes/times are unchanged. M27 remains paused. The fresh
controlled repeat subsequently completed as recorded above.

Latest operator priority: **usable component scans for chassis layout**, setting
mapper research aside. The [practical workflow proposal](practical-component-scanning.md)
defines a dimension-checked CAD reference as the next milestone. The operator
confirmed the Linux constraint and recalled that using the Android build as a
reference one or two weeks earlier did not resolve reliability. The proposal now
records that historical binary-reference approach and supersedes the suggested
vendor-software detour with a bounded assessment of existing offline fixed-axis
reconstruction. Independently measured indexed views are a conditional, unbuilt
alternative, not a relabeling of the existing estimated-axis path. No new scan,
reconstruction or hardware change occurred; M27 remains paused. All prior
research findings and uncertainty limits remain retained below.

The operator also pointed out that the printed target's actual major dimensions
were already measured. The recovered 2026-08-31 print record has X/Y/Z
91.90/81.85/55.85 mm and pocket depth 12.10 mm, against nominal
92/82/56/12 mm. The existing target gate already applies the predeclared axis
corrections. The practical proposal now links those exact records, recognizes
them as usable comparative evidence without claiming full print metrology, and
starts from prior captures/comparisons rather than asking for duplicate setup.
Read-only inspection confirms FreeCAD 1.1.3 is installed as a system Flatpak;
it was not launched. The proposal also corrects an old reporting misconception:
rigid CAD placement cannot absorb scale error, although partial-surface scoring
does not replace explicit dimension checks. Frozen code and old evidence are
unchanged; no new acquisition or reconstruction has run.

The next-step review recovered the 190-frame measured-target capture at
`/home/ben/scans/sweep/20260904-020845`. The practical plan now proposes one
fixed-setting reproduction of its documented positive fixed-axis CAD result,
then explicit dimension/coverage checks and a controlled physical repeat. It
does not mistake the existing chain-based reconstruction's internal readiness
report for that CAD result. It also records that generic single-view/fusion
comparisons and the rigid-scale wording correction already appear in the old
target-gate history, with important later withdrawals. No replay was started;
the current tracking corpus still requires its own physical-target association.

Living index requested by the operator on 2026-09-09 UTC. Detailed protocols,
raw evidence, failed attempts and source snapshots remain authoritative; this
file explains their relationships and records open ideas so they are not lost.
Update the status after each completed phase; never edit a sealed evidence root.
The [reproduction guide](reliability-reproduction.md) explains bundle contents,
safe fresh-output reruns, absolute-path dependencies and local backup limits.

The operator subsequently clarified that transferable research for the mapper
also has value, even without an immediate MINI fix. The
[mapper transfer synthesis](reliability-mapper-transfer.md) separates demonstrated
local findings, proposed reusable methods, sensor-specific settings and untested
generalizations. It supplements this full record rather than replacing it.
This discussion does not resume the paused experiment batch.
The subsequent sibling-repository inspection narrows that transfer: Mapper's
planned LiDAR/camera/GNSS/IMU rig and current GNSS-based pose estimator differ
from the MINI's depth-only tracking. Its course-test targets are provisional
decimetre-scale metrics, not a fine-object metrology claim. The transfer synthesis
records exact source identities, relevant timing/heading risks, manufacturer
references and which MINI-specific work has little direct mapper relevance.

## Current checkpoint: 2026-09-09 16:18 UTC — paused for discussion

The operator requested a progress-versus-churn and hardware assessment. Work is
paused; see the [discussion checkpoint](reliability-discussion-checkpoint.md)
for exact resumable process/source/output state and the proposed causal A/B.
M27's controller is suspended, with 37 scored cells and 39 completed child reports
out of eighty planned; the two in-flight children have exited. No final M27
audit exists. All twenty observation controls and ten original parents are exact.
No queued trial will start while the controller remains suspended.

The [conditional Stage B wrapper](reliability-organized-model-stage-b.md) passes
113 focused / 2,006 selected tests, but no candidate is bound or replayed. Its
CPU inventory was intentionally interrupted after 7,011 raw inputs. All 283
recorded incompatibilities concern dense preparation exceeding an approximate
1,800-point target that the new adapter incorrectly treats as a hard cap; they
are not evidence of scanner damage. The campaign helper is an unfinished draft.
The [association research notes](reliability-association-ideas.md) preserve
separate hypotheses, not an implemented or qualified alternative.

M1–M26 are complete: **695 changed trajectories, 139 whole configurations,
14,595 required/checkable raw pairs, 134 exact observation pilots, one Stage A
survivor**. **Ordinary-900 / centered / reverse-forward clears all five cases
and all 105 fixed checks**, accepting all 2,524 original inputs. Full-phase
and cumulative audits confirm the result. Its narrowest good-pair regression
margin is about 0.1074 percentage points; this is not release qualification.
Four Stage B routing controls reproduce exactly. All six changed longer/handheld
replays finish and the 04:49:32 audit exactly reproduces all 480 fixed scores:
**the Stage A survivor is rejected in Stage B**. All four longer cases fail
geometry; paused handheld accepts 186/188, combined 93/303 and stops after
three refusals. No quality-qualified candidate exists. M22 adds no survivor:
dense variants preserve four cases but fail C; ordinary preserves only statics.
M23's six controls reproduce exactly; all twenty changed trajectories complete
and none of four configurations survives. Its all-input acceptance coexists
with severe moving-case geometry errors. M24's 192-cloud/384-field GPU parity
audit passes exactly (441,026 normal estimates, 384 actual dispatches); six
observation replays start at 05:39 UTC and all six reproduce exactly. Twenty
changed M24 trajectories start at 05:45 UTC and complete/reject at 06:12 UTC.
No whole configuration survives; the best pass three cases. Its closeout counts
30,148 completed fits, no refused fits and 20,053 actual GPU field dispatches.
M25's selection/seed combination finishes all forty changed trajectories and
all 840 fixed pair checks. All eight configurations accept every original
input, preserve both stationary cases, and fail all three moving cases. No
survivor is added. All 840 declared interaction contrasts are computable. M22's
728-field larger-model GPU parity audit passes exactly, including 190 fields
above 2,048 points. Conditional M25 Stage B infrastructure has 63 focused /
1,370 selected tests, clean lint and cold typing of 128 files, but no bound
candidate or recorded qualification. M26's bounded depth median has 95 focused
and 1,465 selected tests passing, clean lint and cold typing of 130 files. The
first run's two private-import typing errors, correction and repeat are retained.
Its 173 executing files and 256 verification files are frozen at 07:23 UTC.
All twelve GPU observation controls reproduce exactly. Forty changed M26
trajectories start at 07:36:41 UTC, with at most two GPU children. The completed
08:32:08 closeout confirms all 840 fixed pairs checkable and no whole survivor.
Every original input is accepted; both four-case configurations still have
severe remaining failures. All 2,524 raw frames reproduce identical derived
hashes and statistics across variants, with valid/band membership exact. These
are selected suites, not the entire repository or accuracy qualification.

No production tracking algorithm, calibration, firmware, driver, dependency,
scanner setting or experimental mesh has been changed during this mechanism
continuation. Existing owner changes are preserved. The five development cases
are reused, not independent validation. Complete frame acceptance often coexists
with severe geometric failure. Nothing is promoted merely for accepting frames.

## Experiment ledger

M1–M15 details and comparisons: [mechanism campaign](reliability-mechanisms.md).
Each row below refers to its entire declared roster, not every conceivable
variant of the named mathematical method.

| Phase | Mechanism | Completed conclusion |
| --- | --- | --- |
| M1 | Centered finite update | Some stationary improvements; no whole survivor |
| M2 | Symmetric source/target-normal objective | No whole survivor |
| M3 | Planarity, directional balance, combined weighting | No whole survivor; not residual weighting |
| M4 | Measured contributor instead of voxel centroid | No whole survivor |
| M5 | Frozen-metric covariance step | No whole survivor; not a full GICP optimizer |
| M6 | Reciprocal correspondence filter | No whole survivor; moving failures |
| M7 | Dense source and retained clouds | Original grid passes four cases, regresses on static B |
| M8 | Stratified model-frame sampling | No whole survivor |
| M9 | Dense source plus stratified model | Centered grid passes four cases, regresses on static B |
| M10 | Native-pixel model sampling | No whole survivor |
| M11 | Voxelize model union before budgeting | Dense variants preserve A/B/A1/A2 but fail C |
| M12 | Separate source and retained capacities | No whole survivor; static B remains binding |
| M13 | Twenty-neighbor normals | Dense variants preserve moving cases but regress on static B |
| M14 | Exact-count stable retained-pixel sampling | No whole survivor despite much more stable membership |
| M15 | Capacity crossed with existing pose-settling stop | No whole survivor |
| M16 | Rotation-aware covariance descent | Every completed step descends; trajectories still fail |
| M17 | Voxel-first model plus twenty-neighbor normals | Complementary individual benefits do not survive combination |
| M18 | Fixed coarse-to-fine voxel pyramid | Ordinary fails C; dense regresses on good controls |
| M19 | Source-normal eligibility / 45-degree compatibility | No whole survivor; two dense variants pass 4/5 but regress on static B |
| M20 | Huber / Cauchy residual weighting | All eight preserve statics, none preserves C; no whole survivor |
| M21 | Reverse fit seeds unchanged forward fit | One Stage A survivor; complete six-case Stage B rejects it |
| M22 | Double ordinary model budget; explicit larger-field GPU route | Complete/rejected: dense passes four cases, fails C |
| M23 | Plane residual plus fixed 0.1-weight 3-D residual | Complete/rejected; all moving cases fail at least one original guard |
| M24 | Tapered local covariance on twenty-neighbor support | Complete/rejected; exact GPU parity/controls, at best three cases pass |
| M25 | Reverse initialization crossed with agreeing-candidate preference | Complete/rejected; eight configurations preserve only stationary cases |
| M26 | Bounded image-domain median crossed with reverse initialization | Complete/rejected; twelve exact controls, forty changed trajectories; at best four cases pass |

Separate records: [M17](reliability-voxel-normal-combination.md),
[M18](reliability-multiscale.md), [M19](reliability-normal-compatibility.md),
[M20](reliability-robust-residual.md), [M21](reliability-reverse-seed.md).
The subsequent [M22 model-capacity test](reliability-model-capacity.md) has a
separate protocol and evidence root and is included in completed totals.
The [M23 mixed-metric test](reliability-mixed-metric.md) is complete and counted,
not a measured uncertainty model or accuracy-qualified result.
The [M24 tapered-normal test](reliability-normal-taper.md) is also source-frozen,
with real GPU parity, controls and all changed trajectories complete. M21's complete
Stage B rejection is now resolved; any new Stage A survivor still takes priority.
The [M21 Stage B record](reliability-reverse-seed-stage-b.md) preserves the
longer/handheld protocol, capture-format controls and separate geometry limits.
The [M25 selection combination](reliability-seed-selection.md) records the next
closed hypothesis, 170-file executing freeze and 253-file verification freeze.
The [M26 depth-preparation combination](reliability-depth-median.md) is complete
and included in the M1–M26 aggregate. Its
original raw scoring and membership-preservation rules remain binding.

At 07:39 UTC, a separate [organized-normal review](reliability-organized-normal-review.md)
is declared: the complete existing 192-cloud raw roster, original voxel
centroids and representative measured pixels, with no pose fit, GPU context or
tracking substitution. It compares immediate image-neighbor cross products
with both original PCA fields while counting unavailable directions. Its 79
focused and 1,544 selected checks pass, with clean lint and cold typing of 131
files. The source-recorded CPU review starts at 07:47 UTC. This is not M27 or
a new scan qualification; neither direction is assumed physical truth.
It completes all 192 clouds / 220,513 normal occurrences, and the 07:49:44
independent scalar audit agrees within 2.23e-16 per direction component. The
separately declared stationary follow-up completes eighty repeated-capacity
frame pairs at 07:52:44. On the same jointly available pixels, image directions
vary by median 6.26–7.54 degrees versus 0.64–1.23 for original PCA fields.
This argues against the immediate unsmoothed substitution, not against all
organized or smoothed methods. No new tracking mechanism or normal is installed.

A separate official-package static review starts at 07:58 under
`20260909-official-package-review/`. The directly manufacturer-linked original-
MINI-compatible Revo Scan 5.5.5.2012 Windows installer downloads completely:
682,230,392 bytes, SHA-256
`b7c632591926d6ac80693ba5255c2855a868497ffe1e31993fe61f961d9244e4`.
No downloaded code is run, installed or loaded. Existing 7-Zip only lists the
PE wrapper; its 64-byte Inno loader resource has a verified revision-2 CRC
and identifies metadata as Inno Setup 6.5.2. The 93,073-byte metadata interval
decodes completely into two blocks (425,606 and 126,380 bytes), with all block
and chunk CRCs verified against the exact-version format. Header counts report
1,422 files / 1,420 locations; five selected camera/configuration records map
unambiguously. The solid stream is LZMA1 with an 8-MiB dictionary, not LZMA2;
the initial unsupported-method refusal and corrected parser are retained.
The completed [package review](firmware-package-review-20260909.md) walks all
1,422 file entries and extracts five preselected files with exact original
SHA-256 matches. No firmware-image filename is found; embedded or remote payloads
are not excluded. All 79 final decoder tests and the 12:17:54 static audit pass.
Calibration-pattern JSON is not factory calibration or a tracking filter.
Static x64 traces and the preserved older Linux SDK establish a real predicate
difference: the newer Windows library restricts the 2.9.43 pattern-5/6 check to
POP 2; the older Linux library applies it to all otherwise eligible models.
This MINI maps to ID 12. Host permission is not device support, and no image,
firmware update, runtime pattern change or reliability claim follows.

At 12:12 UTC a separate `20260909-fringe-capability-review/` begins with a
173-source-frozen, closed scanner health/configuration inventory. It completes
in 1.62 seconds with the same firmware/intrinsics, baseline 20000/20000/0 and
gain 24. CPU/NPU temperatures are 58.0/57.3 initially, 58.0/57.7 C finally.
Six source-recorded nonrecursive directory observations complete at 12:15:46,
with baseline unchanged and read selectors cleared. The existing getter/log
protocol creates bounded uniquely named RAM scratch files; this is no persistent
device write, not literally zero RAM writes. No stream, mode or timing change.
The three-path follow-up and a four-path target catalog complete. A fixed
sixteen-file read plus a duplicate firmware read completes at 12:25:41 UTC;
both 757,024-byte PREISP copies match exactly. Calibration and startup-script
bytes are also preserved. CPU/NPU reach only 58.9/58.2 C during these reads.
The [device review](firmware-device-review-20260909.md) records the catalogs,
their tests, and the additive container parser. All fourteen records parse;
the failed initial header-overlap assumption and its source/tests are retained.
Strings locate timing and fringe controls, but do not establish mode support.
The YAML's serial mismatch was already noted by existing code. The completed
21-test-parser numerical comparison finds all four existing binary fixtures
exactly match fresh reads, and Pl/LR-P/LR-Q exactly match the corresponding
YAML rectified matrices after float32 rounding. This is numerical consistency,
not authenticated metadata or physical accuracy; no correction is inferred.
All nine CRC-flagged firmware sections subsequently match one predeclared CRC
scheme, independently rechecked with the standard library. The 12:53:31 audit
passes all source/byte checks and exact whole-file section coverage. Current-
mode readback, restoration and executable branch semantics remain unresolved;
no faster-mode trial or flash is justified by these static findings alone.
The device review seals at 12:56:59 (334 files / 7,563,546 bytes), manifest
`b233bd2d1329f39ba8a887fdf6f1bbfaaffb5777ba0a7f0d3698e96d672eab66`;
the separate organized-plane root independently verifies it at 12:58:42.

A [larger image-local plane review](reliability-organized-plane-review.md)
is declared before implementation: all 192 original clouds, three fixed window
sizes and all eighty original stationary comparisons. Its first 36 tests and
expanded 73 tests pass, followed by 1,617 selected tests and typing of 132 files.
The 175-executing/218-saved-source run completes all 192 clouds / 661,539
window occurrences at 13:06:34. Independent direct-window checks cover every
support mask; 30,752 sampled scalar/SVD fits agree within declared tolerances.
The eighty-pair comparison retains all 81,587 original common-pixel occurrences.
Radius 2 remains noisier than PCA10. Radius 5 improves same-subset medians
against PCA10, while radius 10 reaches about 0.19 degrees versus 0.41–0.55
for PCA20 but has only 30.3% overall estimate availability. Stability is not
physical normal truth. A separately declared four-profile hybrid review tests
fixed weights 1/0.5 for radii 5/10 while preserving original PCA10 availability
and explicitly counting method switches. Its 39 focused tests pass; the
expanded selected suite passes 1,656 tests, clean lint and typing of 133 files.
Hybrid numerical work completes at 13:20:42: every profile improves aggregate
median/p95 in all four strata, but has worse p95 on its hard-switch subset.
The 326,348 repeated availability checks pass exactly. A separate fixed smooth
support/RMS/angle-taper protocol is declared before implementation to probe
those transitions. The confidence helper's first 36 checks retain one failed
malformed-input case, then pass after explicit status/direction consistency
validation. The first broad suite passes 1,692 runtime tests but retains four
RMS typing errors and is correctly labelled verification_failed. After that
run exits, a validation-only type-narrowing correction passes focused typing
and all 36 checks; the separate broad rerun passes all 1,692 tests, clean lint
and typing of 134 files. Confidence analysis completes at 13:34:48. All profiles
improve aggregate medians with lower/equal p95; the large additional switch
tails mostly return to the original level, but small switch regressions remain.
The 13:37:51 independent audit reconstructs all 326,348 profile/pair occurrences
for both combinations, checks 652,696 endpoint weights and reproduces 256
quantile arrays exactly. Maximum angular discrepancy is under 1e-9 degrees.
Transport into the fused multi-frame model and reverse-fit role binding are
the next separate implementation questions. This fits no pose, alters no raw depth
and is not M27 or a tracking qualification. A 13:19:37 supplement preserves
217 inherited context files, giving 435 distinct plane-review source/context
files, plus four separately retained hybrid/confidence helper/test files, for
439 total. The complete review seals 690 files / 756,138,314 B at 13:47:26;
outside verification confirms exact inventory and all hashes at 13:55:34.

The subsequent [raw-backed model preflight](reliability-organized-model-preflight.md)
finds a coordinate-only normal-cache hazard: equal model/current points can
have different unsampled raw-image support. Its separate pure dispatcher checks
explicit forward/reverse roles before hybrid reuse. Ownership follows accepted
frame identity, exact native pixels and raw hashes, not coordinate equality.
Initial transport/binding checks pass 83 tests, then 1,775 selected tests,
clean lint and cold typing of 136 files. All 443 source/context files are frozen.
At 14:11:43, forty original R1 model states reproduce all centroids, contributor
identities and 55,897 saved GPU-PCA estimates exactly. All 160 fixed combinations
and 223,588 availability checks pass; independent blend differences are below
1.05e-15 per direction component. This uses 86 raw frames and fits no new pose.
The 14:13:30 independent scalar/SVD audit verifies 140,734 native point identities,
samples 8,518 windows and compares 4,174 usable directions within tolerance.
Subsequent mutable-nesting tests expose four missing constructor guards. Their
failed tests and original frozen source remain intact; explicit tuple validation
makes all 87 focused tests pass. The hardened broad run passes 1,779 runtime
tests/typing but retains one new test-regex lint issue. After its process exits,
only the raw-regex marker changes; final selected verification and exact saved-
result reproduction gate the next scoped adapter. No M27 trajectory is counted.
The final run passes all 1,779 tests, lint and cold typing of 136 files. At
14:22:49 the hardened helpers exactly reproduce all 172 native fields and 160
model combinations, including all 223,588 estimates and companion records.
That completes the pure-model preflight; actual adapter/trajectory controls are
still required. Source versions and actual failed tests/checks remain preserved.

The separate [M27 scoped model adapter](reliability-organized-model.md) passes
114 focused / 1,893 selected tests, clean lint and cold typing of 138 files.
Its 447 context/source files and 181 executing identities are frozen at 14:43:52.
Synthetic original-runner checks use real raw preparation but explicit CPU-normal,
loader and scoring doubles; they are not recorded GPU or geometry qualifications.
The planned twenty GPU observation controls precede eighty changed trajectories
(sixteen global radius/weight/capacity/seed combinations). Prelaunch lookup stops:
M21 never recorded forward-centered seed traces. M1/M7 contain the matching
original trajectories, so a separate ten-parent, source-frozen prerequisite
begins around 14:55. It uses the unmodified M21 CLI, compares exact old trajectories
and recomputes both old/new scores on the complete fixed roster. No changed M27
normal trajectory is counted yet, and no scanner acquisition or setting changed.
All ten unmodified parents subsequently reproduce exactly; the 15:05:52
independent audit verifies 5,048 input occurrences and recomputes 420 old/new
pair scores across 210 fixed identities. The full 100-cell M27 plan freezes
at 15:08 and its twenty observation controls start. A separately source-bound
independent control audit is mandatory before any of the eighty changed runs.
At 15:24, fifteen controls have completed exactly and five remain in progress.
Additive predeclared CPU reconstruction covers first/middle/last model states
from every control before applying the same rule to the changed matrix. Full
four-factor descriptive interactions will use all sixteen configurations and
all 105 pair identities; neither sampled normal reproduction nor an isolated
contrast substitutes for the unchanged whole-configuration geometric gate.

Partial Stage B checkpoint, 04:41 UTC: development extended and slow D finish
with complete acceptance but fail fixed geometry, minima 35.011% and 56.898%.
The six-case roster still runs. Development's worst pair is adjacent raw inputs
520/521 (original 99.231%, changed 35.011%), so the problem is not exclusively
long-distance loop closure. All saved alternatives are being audited at the
original fixed pairs; no alternate pose or selection policy has been installed.

At 04:43 UTC the complete 103-pair development saved-alternative audit shows
all ten failing local pairs have an accepted adjacent candidate preserving the
original guard. Only one of seven failing revisit endpoints has any preserving
alternative; the two revisit record kinds share endpoints. This suggests a
future reverse-seed/selection combination, not a ready policy or proof it will
repair feedback history. M22/M23 remain deferred; additive continuation drivers
are prepared to require a complete, exact and explicitly rejected Stage B audit
before treating M21 as resolved. Original drivers/protocols remain untouched.

## Binding rules and deferred qualification

Every whole configuration must accept every required input and pass all five
recordings, every frozen raw pair at least 90% within 1.5 nominal mm, with no
originally good control-pair loss over one percentage point. Static A is the
adverse case: it still needs 90%, but has no one-point regression guard.
Missing checks/poses remain missing, never imputed or silently excluded.
No per-recording method switch, newly selected best pair, tightened crop or
changed threshold substitutes for a whole pass.

The [roadmap](reliability-roadmap.md) still requires longer/handheld Stage B,
fresh physical fixed/handheld loops, coverage, live integration and scanner
thermal endurance. A Stage A survivor takes priority over later new ideas.
The available recordings do not independently bind the current printed object
to a CAD/STL reference; no CAD-based accuracy claim is justified yet.

## Findings worth carrying forward

The [six-capture signal/timing audit](reliability-signal-timing.md), completed
around 06:01 UTC, inventories all Stage B capture folders and proves all nine
inspection PNGs are exact depth renders. No independent RGB/IR or inertial
stream is identified there. The 5,139 instrumented inputs map exactly to saved
kernel/host records, but the alternating short/long intervals have opposite
phases in kernel versus dequeue-time timestamps. No exposure semantics or
correction is inferred from the driver's SOE label. The actual tracker consumes
offered frames, not elapsed seconds; a time-aware prior would be a new algorithm.
The two legacy project time arrays are preserved but absent from the original
replay input manifest; paused handheld has none. These findings do not change
any frozen replay, score, model or scanner setting.

### Hardware, timing and earlier live processing

The [2026-09-09 public firmware-path recheck](firmware-public-review-20260909.md)
records current manufacturer links, original MINI versus MINI 2 distinctions,
retrieval limitations and the still-open recovery/calibration requirements.
It is read-only and does not establish this serial's available update version.

These are historical measured states, not fresh device readbacks during this
replay-only continuation. The exact-unit record names serial
`A22704522A6F00A64`, firmware `v2.8.7.20220610`, algorithm `v7.5.7.0426`.

- The scanner delivered approximately 13 raw depth frames/s over the same
  measured 480 Mb/s USB2 link. USB3 negotiation is not required to explain the
  measured rate improvement. Original 20/20 ms produces about 4.5 raw fps;
  ordinary V4L2 FPS requests did not change that. Exact-unit volatile exposure/
  frame-time controls did. [Timing/firmware evidence](scanner-fps-investigation.md).
- The stopped-table comparison supersedes the early optimistic moving-view
  retention estimate: 7–10 ms persistently lost about 17–24% of baseline-stable
  pixels in that scene. A later fixed 20/14/14/20 ms block gets about 6.59 fps
  at 14 ms but only 73.97–74.74% edge retention, versus a 95% requirement.
  The mask includes support; this is not “25% of the printed part lost.”
  Original 20 ms remains the default. [Stopped/moving results](handheld-reliability-results.md).
- Exposure/gain also affect face rippling: reducing intensity improved some
  median-map plane residuals, while other views lost depth. Neither the
  original setting nor a favorable faster block is calibrated shape truth.
  [Full earlier validation](scanner-validation-results.md).
- GPU FP64 normal eigensolves are integrated in the GUI worker; neighborhoods,
  correspondence search and other stages remain CPU work. Short awake-display
  tests applied previews at about 10.4 fps; the older five-minute GUI run was
  9.01 fps, with missing deliveries and later-proven geometric drift. Model
  refresh, delivered frames and correct accumulated shape are different metrics.
  Experimental meshing stays disabled. [Live processing](live-gpu-throughput.md).
- A **15-minute raw-only** 7 ms soak already exists: 11,650 unique complete
  frames, 12.950 fps, no short/sequence losses, CPU/NPU peaks 67.8/67.0°C,
  verified 20 ms restoration. This is not the pending quality-qualified
  candidate/GUI/handheld endurance or optical dimensional-drift qualification.
  The final temperature slopes remained positive, so no indefinite plateau.
  [P5 record](scanner-validation-results.md#p5-15-minute-maximum-tested-raw-acquisition-endurance).
- Scanner CPU configuration reports passive trips at 75/85°C and critical
  115°C; no independent NPU limit was established. Do not assume 90°C is a
  useful continuous operating target. The test policy's 80°C stop is not a
  manufacturer optical safety rating. Host GPU temperatures are separate.
- Recovered controller disassembly and logs explain a routine delayed gadget
  stop/restart; they do not establish the cause of every unrecoverable USB
  fault. A controller binary and 40-byte intrinsics file are not a full firmware/
  calibration backup. Flashing remains gated by an authenticated exact-model
  image, complete calibration preservation and supported recovery, not merely
  permission to flash. [Firmware gate](scanner-validation-results.md#p6-firmware-evidence-and-flash-gate).

### Tracking, data and evidence limits

- The [calibration-semantics review](reliability-calibration-semantics.md) separates
  the August caliper-based unit result from still-unresolved arbitrary-view
  optical accuracy. The pinned vendor header uses the same axial-depth/pinhole
  conversion. All 1,038,206 nonzero points in fifteen preserved metric fixtures
  match independent float64 arithmetic; explicit float32 rounding differs by
  at most 0.00001936 nominal mm. This is not SDK execution or a calibration fix.
  A separately preserved rejected fixture lacks the historical manifest binding.
- The longer D recording accepts all 1,609 inputs but fails geometry. Direct
  same-view fits recover every tested D pair from both prescribed starts. This
  implicates accumulated history/reference consistency; it does not establish
  correct poses between revisits. [D evidence](reliability-slow-turntable.md).
- At all 15 bad saved D states, none of the already-saved alternatives reaches
  90%. Simply selecting another saved candidate cannot repair those states.
  A different earlier history has not been ruled out.
- Native-frame model normals alone fail good A1/A2 regression guards. Their
  small initial differences propagate into different keyframes and references.
  [R1/R2 evidence](reliability-reference-model.md).
- Prior global loop/graph experiments L1/L2/L3/G1/G2 are rejected in their own
  records. A good fitted closure did not reliably preserve other checked views.
  They are not a general disproof of all global registration methods.
- The F1/M1/M7 grid prediction audit finds no active motion-prior seed and no
  travel-gate decision relying on its allowance in the twenty inspected runs.
  Non-anchor frames have locally accepted adjacent edges. A no-prior ablation
  is therefore low priority here, not ruled out for future handheld recordings.
- Stable pixel membership is not enough: matched-count retention increases
  adjacent membership overlap from roughly 49% to 97%, yet M14 fails geometry.
- M16 corrects the covariance objective's rotated-normal gradient and records
  110,226 completed nonincreasing steps, with zero failed steps. No trajectory
  configuration survives. Descent is not correspondence or pose correctness.
- M20 likewise records 142,921 strictly decreasing frozen robust-cost steps,
  with zero failed steps and no activation of its raw-code scale floor. Every
  whole configuration still fails; arithmetic descent alone is not the missing
  reliability guarantee. All original fixed pairs remain in the results.
- Host arrival spacing is not exposure spacing. The timestamp is taken after
  V4L2 dequeue. Alternating short/long deliveries cannot justify rescaling
  motion priors as though they were calibrated camera timestamps.
- Static delivery-parity maps have tiny typical offsets and cross-capture
  correlation only 0.0174. Delivery parity is not verified DSP phase. No
  general alternating-depth correction was justified or installed.
- Normal fields match the CPU reference exactly on the fixed tested GPU
  schedules: 424 ten-neighbor fields, 232 twenty-neighbor fields, 576
  pyramid-level fields and 728 larger-model fields. This verifies those
  computations, not physical normals or trajectory quality.
- At fixed saved histories, M22's doubled model allowance leaves all eighty
  ordinary stationary model point hashes identical, but raises dense static-B
  median occupied voxels from 1,227 to about 1,602. It also changes pooled
  ten-neighbor spatial support. This is a count/normal counterfactual, not
  coverage truth or a fitted larger-model trajectory.
- All 480 original long/handheld historical placement checks reproduce exactly.
  A bounded Stage B wrapper now handles the two capture formats explicitly:
  three legacy projects and three modern raw captures. Direct/wrapped GPU
  controls match exactly. The changed six-case qualification runs; no longer
  geometry result is claimed before its full audit.
- M21 demonstrates a useful combination on the fixed development cases:
  reverse initialization plus a centered update passes where the corresponding
  single changes do not. Dense capacity does not preserve that benefit. This
  supports testing combinations, not assuming their benefits add generally.

## Investigated but not implemented: 2026-09-09 02:24 UTC

The apparent 2,048-point normal limit is an ICP routing/cache-eligibility
condition in `src/scanner3d/icp_plane.py`, not a demonstrated GPU hardware limit.
Above it, the fitter calls CPU `estimate_normals` directly, even with an active
GPU context. The existing GPU eigensolve takes a batch-sized buffer and dispatch;
neighborhoods/covariances remain CPU-computed. GPU activation messages alone
would not verify GPU use for a future larger-target experiment.

This suggests a distinct model-capacity test, but it is NOT declared or run yet.
Before any such trial: check historical budget sweeps, explicitly bound the
larger working/cache memory, validate CPU/GPU parity on larger actual fields,
verify per-fit provider dispatch, preserve truthful policy/report metadata and
all gates, and isolate any routing change to the diagnostic process. Current
M19/M20/M21 frozen experiments retain the original 1,800-point model and are
unaffected. No shader, cache limit, policy or production code was changed here.

Follow-up at 02:45 UTC: M22 is now declared and diagnostic implementation is
under test. A real numerical fixture above 2,048 points confirms provider
bypass under the old routing and provider use under the scoped 4,096 limit,
with identical full results for both updates. This fixture uses a CPU-reference
provider, not real GPU parity. No recorded-scan M22 trajectory has run.

Scene framing remains another limitation: the largest depth component can
include support, and depth-only images cannot prove semantic object isolation.
No support/object crop or multi-body correction has been applied. Further
diagnosis must not mistake a selective crop for preserved object coverage.

Additional prospective discriminator, recorded 06:28 UTC: independently bind
the actual printed target to its STL/CAD and identify object versus support
before re-evaluating what lost depth is meaningful. The earlier 17–24% missing
baseline-stable pixels and failed edge-retention test include support; neither
is a measured object-loss percentage. A separately acquired empty-support
reference with fixed camera, or a verified target reference, could help diagnose
this ambiguity, but neither exists for the current closed corpus. It would
require deliberate physical setup/target identity, and a predeclared protocol,
not a post-hoc crop or a waiver of any M1–M25 guard. This is recorded as future
work, not a request to the absent operator or an experiment already performed.

### Untried idea: continuously tapered normal support

Recorded 03:39 UTC; **not declared, implemented or tested**. M13 changes the
number of neighbors, while M3 weights solver correspondences. Neither tests
a smoothly distance-weighted covariance for each normal. A distinct possibility
is tapering contributions to zero at the outer neighbor distance, so a point
entering/leaving a nearest-neighbor patch has vanishing boundary weight.
This could reduce abrupt patch-membership effects; it could also reduce useful
support, amplify noisy local geometry or preserve the same wrong history.
No causal claim follows from that mathematical motivation.

Before a trajectory experiment: define one closed kernel/support rule, retain
actual positive-weight/effective counts, preserve sparse/degenerate refusals,
verify independent covariance/eigen arithmetic and real GPU fields, require
exact observation controls, and compare against the matching unweighted
neighborhood. Do not describe weighting as measured sensor uncertainty or
choose a best support rule separately for each recording. M20–M23 remain
unchanged, and any survivor takes longer/handheld qualification priority.

An inertial workaround is not presently backed by an available measurement:
the [earlier exact-unit probe](illumination-actuators.md) found only the SAR ADC
IIO device, not the firmware-expected accelerometer/gyro/proximity endpoint.
That is evidence of no exposed/probed endpoint in that observation, not a
teardown proof about every component. No IMU stream or calibrated exposure
timestamps exist in this Stage A evidence to fuse into its tracking.

Follow-up at 03:44 UTC: M24 is now declared in
`20260909-reliability-normal-taper/M24-PROTOCOL.md`. It fixes the existing
twenty-neighbor support and one squared taper, with matching M13 controls.
Implementation/parity/trajectory gates are still pending; the preceding
untried note is preserved as the origin of the hypothesis.

### Newly motivated combination: reverse initialization and candidate preference

Recorded 04:46 UTC; **not declared, implemented or replay-tested**. The complete
M21 development Stage B saved-candidate audit differs from the earlier D oracle:
all ten failing fixed-local comparisons retain an accepted adjacent candidate
that preserves their original good-pair guard. Model-first selection admits
different local corrections into the history. Most late revisits have no good
saved alternative, so changing only their final selection is insufficient.

A distinct closed-loop test could cross forward/reverse initialization with
the original model-first versus adjacent-first preference *only inside the
original agreeing candidate cluster*. Keep candidate computation, agreement,
singularity/travel/overlap/residual rules, source/model budgets, original fixed
pairs and frame acceptance unchanged. Compare whole trajectories and the actual
seed-by-preference interaction; never choose a candidate using future raw checks.

This might preserve local motion while reducing pooled-history corrections; it
might instead accumulate ordinary pairwise drift or lose benefits of broad
reference geometry. Earlier candidate-budget/keyframe-first experiments and
saved-state oracles are not this combination. Inspect those records first;
require exact baseline routing and full Stage A/B qualification. No preference,
threshold or production policy is changed by this hypothesis note.

Follow-up at 05:04 UTC: M25 is declared at 04:56 in
`20260909-reliability-seed-selection/M25-PROTOCOL.md`, before implementation.
Its closed roster has twelve original-preference routing controls and forty
changed adjacent-first cells, crossing both seeds, capacities and updates.
Original candidate computation and all agreement/refusal/rescue gates remain;
only preference inside the original agreed cluster changes. Initial 33 scope,
arbitration and real-tracker composition tests pass; adapter testing continues.
M22–M24 and any unresolved survivor retain execution priority. The original
hypothesis note remains as its origin, not a claim that the idea is still absent.

## Durable evidence map

All directories below are under `/home/ben/scans/diagnostics/`. A sealed root
is immutable: the manifest excludes itself, and a second, outside-root check
verifies exact inventory and every file hash. Source snapshots, protocols,
plans, raw-input identities, full reports, logs, failed attempts and audits are
retained together. Repository documents remain living summaries.

| Evidence directory | Scope | State / manifest SHA-256 |
| --- | --- | --- |
| `20260908-reliability-mechanisms` | M1–M15 execution; M16 preparation | Sealed, 5,235 files; `2f17db2b758157a93deb78e060a96ace1384d40795dcc242609bb9df7b7d0823` |
| `20260909-reliability-covariance-descent` | M16 execution | Sealed, 531 files; `3c8a03abf9943a3ca715ecbd2604285d6a016384c9ba340200a9c1b5bd41a15c` |
| `20260909-reliability-voxel-normal-combination` | M17, delivery-parity audit | Sealed, 577 files; `e71e8baf60e3446f5c08abfb1c9e58eb01cb55a0c21bf0b979e2f6b91d5aaf77` |
| `20260909-reliability-multiscale` | M18, pyramid GPU parity | Sealed, 551 files; `4f91f0694cabd6f4c0af2807f56d92c2c9f0618b15f1a938d6563e8e2598b64f` |
| `20260909-reliability-normal-compatibility` | M19; outside-root M17/M18 seal verification | Sealed, 654 files; `9b8582c9f126b6918ca410af4a6957d76ec69290352a53769bfb8f9d2ddf5210` |
| `20260909-reliability-robust-residual` | M20 | Sealed, 652 files; `c4a769062d38e943e6bffdf4fbb969caf7dfd666554c06cc6426bc9b645a52b6` |
| `20260909-reliability-reverse-seed` | M21 | Sealed, 561 files; `9dfd71d299e7f7e46b9073a79375983f32ff877b6f28c2c4111f133cd5004004` |
| `20260909-reliability-model-capacity` | M22 | Sealed, 581 files; `494456b6bb44cee0b2f523ee7ae3f595f13e985f4fa7b9348e26e35519206167` |
| `20260909-reliability-mixed-metric` | M23 | Sealed, 582 files; `f118825b03843bd28e433c8f367fe7b4b3d925ca90642a826294c36fa20a09e3` |
| `20260909-reliability-normal-taper` | M24 | Sealed, 606 files; `3d6d227b696692506722ef4b9c64de48cc47d8928aabb60885ae66a7be628ddf` |
| `20260909-reliability-reverse-seed-stage-b` | M21 survivor qualification | Sealed, 516 files; `1b9daa8e20bd046d28a5be8ff569e916dd32f4abf5cd7cdec97b323ddceee38c` |
| `20260909-reliability-seed-selection` | M25 | Sealed, 761 files; `4af8bbcae42d5ff473b861f203d1461afde520e5d325c572ca844aaa746fe69d` |
| `20260909-reliability-seed-selection-stage-b` | Conditional infrastructure only | Sealed, 437 files; `536cb9880555d8dee8aac4dcac0418816b4a7de2e0dabbbd879e474e9ae724c0` |
| `20260909-calibration-semantics-review` | Published conversion, historical metric preservation, arithmetic | Sealed, 61 files; `a089425ff28903ee32b611c542d94050468541ebf1088578a0c51ab74db51597` |
| `20260909-reliability-depth-median` | M26 | Sealed, 776 files; `39ce482bab20f4aa8e8e149aa917b993c104cafe842ae339e53005daa458165f` |
| `20260909-organized-normal-review` | Raw image-neighbor feasibility only | Sealed, 457 files; `154e6627f123290d30b3307d43ec3593b94cd1244bcf4b0522bc6adf7f147b10` |
| `20260909-official-package-review` | Official desktop installer, static only | Sealed, 121 files; `93dd2082a9ff1d15e447d57b6eb0b6c6ce90f2d4dd36231cc74171a7bd39cb71` |
| `20260909-fringe-capability-review` | Device PREISP/calibration, CRC and consistency audits | Sealed 334 files / 7,563,546 bytes; outside verification exact |
| `20260909-organized-plane-review` | Larger image-local planes and normal combinations | Complete; sealed 690 files / 756,138,314 B, outside verification exact at 13:55:34 |
| `20260909-organized-model-preflight` | Raw-frame ownership, model transport and fit-role checks | Complete; sealed 1,236 files / 209,460,966 B; outside exact at 14:23:55 |
| `20260909-reliability-organized-model` | Scoped organized/PCA model-normal candidate | Paused for operator review: 1,893 selected tests, ten exact parents, twenty exact controls; 37 scored / 39 completed child reports of eighty changed runs; no final audit |
| `20260909-historical-throughput-preservation` | Specifically documented early `/tmp` evidence | Sealed, 684 files; `fbe22cfff0df23537401a96056d89831cdf9b33f20e620f605940c0675b70a5c` |
| `20260909-historical-tracking-preservation` | Earlier GPU/GUI reports and harness | Sealed, 22 files; `04ff7ce3c5ea9d9170d0729855b131ac08aed1346f29d87588b270935d181873` |

The historical preservation copied 677 original files / 338,902,573 bytes,
plus documentation/protocol/inventory records. No selected path was missing;
every copied hash and original size/mtime/hash was checked. Exact sealed
inventory and all hashes were independently verified at 02:55:49 UTC in
`20260909-reliability-model-capacity/historical-preservation-seal-verification.json`.
This closes the two documents' specifically listed `/tmp` retention gap, not
every possible historical artifact gap. Original absolute paths remain in old
JSON; `source-inventory.json` provides the relocation mapping.

Earlier capture/replay roots and their seven verified seals are inventoried by
`20260908-reliability-mechanisms/prior-seals-final-check.json`. The latest
completed aggregate is `20260909-reliability-depth-median/through-m26-audit.json`;
it excludes Stage B and separate descriptive source/normal reviews.

M19's sealed inventory and all hashes are verified at 03:06:46 UTC in
`20260909-reliability-model-capacity/normal-compatibility-seal-verification.json`.
M20 is also sealed; further host snapshots now belong under M21. Never append
the sealed M19/M20 roots. M20's last snapshot at 03:42:46 UTC was GPU
49/50/72°C edge/junction/memory, host CPU package 75°C, no reported alarm.

Follow-up at 04:35 UTC: M21 is also sealed, 561 files / 1,055,471,026 bytes.
Its outside-root verification lives under the separate Stage B root; subsequent
snapshots belong there, not under M21. M21's final host-only snapshot at 04:25:02
was GPU 50/51/73°C, CPU package 71°C, no reported alarm. Scanner temperature
was not measured. A fresh 04:34:45 source-boundary check confirms all 79 original
production files remain unchanged. None of these facts qualifies live scanning.

M21 Stage B is sealed at 04:54:01 UTC: 516 files / 985,742,765 bytes. Its
outside-root verification is under M22 as
`reverse-seed-stage-b-seal-verification.json`. Do not append that sealed root;
future host snapshots now belong under M22. Its four complete saved-candidate
audits cover all 480 fixed records, including all 32 failing fixed-local pairs
with an accepted adjacent alternative preserving the original pair guard.
Most bad long revisits remain wrong across all saved alternatives. The handheld
diagnosis retains every original/changed refusal; four of five changed refusals
coincide with return warnings, one does not. No physical cause is inferred solely
from that co-occurrence.

Storage check at 03:37 UTC: the repository and diagnostics are on the same
3.6-TB filesystem, with about 1.9 TB available. These preserved copies and
hash seals protect provenance and avoid temporary-file expiry; they are not
an off-machine backup or protection against failure of that filesystem.
No external backup destination, upload, commit or push was authorized here.

M22's seal verifies exactly at 05:22:37; M23 is sealed at 05:43:50, 582 files /
858,300,414 bytes, and verifies outside its root at 05:46:04 UTC. Future host,
link and seal-verification outputs now belong under open M24, never M22/M23.
The latest link audit checks 26 documents and 72 local destinations with none
missing; it does not validate heading anchors or fetch twenty remote URLs.

At 05:46 UTC, a wider documentation search identifies additional early
performance references in `docs/tracking-performance.md` still pointing into
`/tmp`. The earlier preservation intentionally covered two different documents;
this is an additional retention task, not a failure of their sealed inventory.
Device-internal temporary paths and example future output names are not host
experiment artifacts. The specifically named early GPU/GUI reports and harness
are being checked before a separate bounded preservation archive is created.

At 05:48:05, that additional copy completes: sixteen files / 31,021,630 bytes,
none missing, all source-before/after and destination hashes exact. Failed,
corrected, refined and precise GPU attempts are all included. The historical
harness is preserved as found; its later mtime is not proof its exact bytes ran
in an earlier measurement. No old report or original temporary file is changed.

That additional root is sealed at 05:48:45 UTC: 22 total files / 31,054,687
bytes including the protocol, inventory and documentation snapshot. A fresh
05:49 storage check still shows about 1.9 TB available on the shared filesystem.

M24 seals at 06:17:35 UTC, 606 files / 690,197,039 bytes. Its exact outside-root
inventory/hash verification completes under M25 at 06:18:59. Subsequent host,
link and seal-check outputs belong under open M25, not M24. At 06:24, all six
ordinary-capacity M25 controls reproduce exactly; the six dense controls are
running. No partial control count is added to completed aggregate totals.

M23's completed step audit also records a useful negative: 73,919/73,920 combined
cost steps are nonincreasing and the remaining increase is only
1.5178830414797062e-17. Severe moving-case failure therefore cannot be attributed
to widespread ascent of this tested combined cost from these records alone.

At 06:04 UTC, `20260909-reliability-normal-taper/reproducibility-environment.json` records
the current Python 3.14.7 / NumPy 2.5.2 / SciPy 1.18.1 runtime, all 25 installed
distribution versions, numerical build configuration, CPU/OS/kernel information,
selected non-secret environment settings and existing Git state. All 168 frozen
executing-source files still match. Exact current project/lockfile/venv/OS
configuration copies are in that root's `environment-config/`. The separate
native-library fingerprint hashes 186 mapped files / 426,652,448 bytes from two
explicitly validated running M24 children; it does not read process memory or
copy/install drivers. This is a current reproducibility aid, not a historical
environment attestation or physical-accuracy certificate. NumPy/SciPy's optional
pretty-output warning did not prompt installing pyyaml. The driver supplies
`QT_QPA_PLATFORM=xcb` to children even though the parent snapshot leaves it unset.

## Methodological corrections and negative results

- Read-only review at 07:37 revisits whether a stronger singularity cutoff is
  justified. The existing [M1 information audit](reliability-mechanisms.md)
  already covers all 106,491 recorded steps; good and failed trajectories have
  overlapping, sometimes oppositely ordered normalized condition ratios.
  [The two-view audit](reliability-view-b.md) likewise finds improved apparent
  stationary placement despite worse conditioning. This is not new numerical
  evidence and does not justify a newly tuned cutoff. Full-rank biased
  correspondences can still produce a wrong pose; no guard is changed.
- At 07:33:47 the host-only snapshot is GPU 49/51/73 degrees C
  edge/junction/memory and CPU package 75 degrees C, no reported alarm.
  The scanner was not operated or temperature-polled. Outside-root checks under
  M26 verify the calibration-review seal at 07:13:08, conditional M25 Stage B
  infrastructure at 07:16:17 and completed M25 at 07:23:54. These are local
  integrity checks, not an off-machine backup.
- M18's frozen pilot driver omitted the protocol's full solver-trace/raw-score
  equality checks. Before any changed M18 trajectory, a separate additive audit
  verified all six pilots and all 23,315 solver records exactly. The original
  driver, plan and pilot reports were not rewritten. Later drivers include
  these comparisons directly.
- The first prediction audit used the wrong literal field name `prior`;
  the failed attempt is retained. The corrected audit uses `motion_prior`.
- M18's first numerical fixtures called a nonexistent `Rigid.translated` helper;
  this was a test-fixture error, corrected to `Rigid.of` before source freeze.
  It was not a failed scan or a changed recording result.
- M23's first expanded iterator test similarly named nonexistent `_pairs`/
  `_trim` helpers; all four failed cases remain in their JUnit report. The
  corrected test observes the actual last solver pair set. Importantly, the
  original iterator's reported plane RMSE uses those frozen pairs after its
  update, not a new nearest-neighbor search at the returned pose. M23 preserves
  that semantics and independently checks the arithmetic; no gate is rewritten.
- The first Stage B saved-candidate audit mistakenly applied the ordinary
  four-candidate bound to recovery-attached decisions. It stopped without final
  JSON. The attempted source and correction note are retained; the read-only
  correction includes the original separately bounded recovery candidates.
  No original scan, fixed score or tracker gate is altered by this correction.
- The physical raw depth-code scale is `0.004999999888241291` in the library's
  nominal coordinate units. Do not substitute the earlier mistaken `0.01`, or
  call a code step a measured sensor noise standard deviation.
- Passing unit tests, exact CPU/GPU arithmetic, or decreasing an optimization
  objective does not establish physically accurate tracking. Concurrent replay
  wall times are not a live GUI/FPS benchmark. These distinctions remain binding
  even when an isolated result looks promising.
