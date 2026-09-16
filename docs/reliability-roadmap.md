# Reliability roadmap and test map — 8 September 2026

Execution records: [first round](reliability-roadmap-results.md),
[registration/thermal continuation](reliability-registration-continuation.md),
[independent face-B diagnostic](reliability-view-b.md),
[slow-turntable follow-up](reliability-slow-turntable.md),
[reference/model normal diagnosis](reliability-reference-model.md),
[the combined-factor experiment](reliability-factorial.md),
and [the solver/model mechanism continuation](reliability-mechanisms.md).
The gates below remain the predeclared plan; experiments are not releases.
The [living learning index](reliability-learning-log.md) records conclusions,
rejected ideas, open questions and evidence locations across all rounds.

## Destination and present state

The destination is **reliable acquisition and reconstruction inside an explicitly
tested operating envelope**, including handheld use. It is not unconditional
tracking on featureless, transparent, reflective or out-of-range surfaces, and
not a claim of the manufacturer's dimensional accuracy without calibration.
Safe, clearly reported refusal outside the envelope is part of reliability;
silently accepting a bent scan is not.

Current evidence: 20/20 ms, gain 24, auto exposure off gives about 4.51 raw fps.
Two recent slow-turntable controls accept all 628 settled frames, with weakest
sampled revisit agreement 96.50% / 98.48%. At 14/14 ms, 6.60 fps costs raw
depth; the extended replay accepts 1,171 frames but its weakest revisit is only
76.45%, versus 99.94% for the same raw views at identity. Handheld move/pause
and adverse moving recordings still fail or drift. GPU normals are active;
extra GPU speed has not solved registration. No experimental live mesh.

Latest follow-up: the six-minute, 20 ms slow-turntable diagnostic retains 1,621
clean raw deliveries and accepts all 1,609 settled replay inputs, but 15 of 40
raw-selected revisits fail the 90% margin; the weakest is 15.02%, despite raw
identity agreement >=98.99%. Direct matching-reference fits succeed on all 40
pairs, but are diagnostic fits, not a corrected or independently validated
trajectory. Mean offline GPU-assisted preparation/tracking is 93.86 ms; the
registration gate, not processing throughput, remains the release blocker.

The subsequent native-frame model-normal experiment also remains rejected:
although A1/A2 accept all 628 inputs and retain minimum revisit scores above
96%, original checked pairs lose 1.299 / 1.657 percentage points, exceeding the
unchanged one-point no-regression guard. Pose differences begin at source 4 and
propagate into changed keyframe history at source 24. No production change was
promoted; a new default replay reproduces all 628 A1 poses and decisions exactly.

The subsequent closed 3 x 2 x 2 combination experiment runs all sixty cells:
twelve sampling/model-normal/convergence configurations on static A/B and
moving A1/A2/C. Combinations produce large matched-pair improvements (rank plus
settling raises one A2 comparison from 13.72% to 95.03%), but no configuration
passes all five controls. That same combination reaches only 16.10% on C;
rank plus native normals plus settling accepts every A1/A2 input while geometry
falls to 30.14% / 8.19%. No candidate advances to the predeclared longer/handheld
stage or live integration. These are configuration-level rejections, not proof
that all combinations of methods must fail. The complete matrix and interaction
evidence are in [the combined-factor record](reliability-factorial.md).

At 2026-09-09 08:32 UTC, the autonomous mechanism continuation has completed
M1–M26: 695 changed development replays, 139 whole configurations, 14,595 fixed
pair checks and 134 exact observation controls. One ordinary/centered
[reverse-seeded configuration](reliability-reverse-seed.md) passes all five
short cases, but its full [six-case longer/handheld qualification](reliability-reverse-seed-stage-b.md)
rejects it. All four long cases fail geometry; the combined handheld replay
stops after three original-gate refusals. No quality-qualified candidate exists.

The [larger model](reliability-model-capacity.md) and
[mixed residual objective](reliability-mixed-metric.md) add no survivor.
[Tapered normal estimation](reliability-normal-taper.md) has exact 384-field
GPU arithmetic and six exact recorded controls; all twenty changed trajectories
complete, but no whole configuration passes. [Reverse initialization crossed with agreeing-candidate selection](reliability-seed-selection.md)
has twelve exact controls and forty completed changed trajectories, but all
eight configurations preserve only stationary cases. It was justified by saved
alternatives at all 32 failed fixed-local Stage B pairs, not a repetition of
the old preference sweeps. Most bad long revisits still have no good alternative
at the saved endpoint; the attempted changed history does not repair it.
The separately declared [bounded depth-median combination](reliability-depth-median.md)
passes its 1,465-test selected verification and has frozen executing sources.
All twelve observation controls reproduce exactly. All forty changed
trajectories complete, preserving every input and all 840 raw comparisons,
but none of eight configurations passes all five cases. No Stage B is earned.
No production change or gate waiver follows from any of these diagnostic tools.
The [living learning index](reliability-learning-log.md) owns later checkpoints,
complete evidence locations, exact controls and deferred qualification work.

The historical 75%-within-1.5-nominal-mm revisit test remains unchanged. A narrow
pass does not meet the stronger **promotion margin below**, declared before new
experiments. Existing scanned scenes and replays are development/regression
data, not blind holdouts. Frame acceptance, accuracy, coverage and raw depth
retention are separate verdicts.

## Required release gates

1. **Evidence integrity and safe acquisition.** Every delivered/selected/refused/
   short frame has explicit provenance; no unreported loss or guessed pose.
   Correct serial/settings; clean cancellation, bounded recovery, verified
   restoration after streaming, no unresolved transport fault. Expected quiet
   stop-time USB re-enumeration is recorded, not disguised as continuity.
2. **Geometric reliability, not just continuity.** On usable fresh validation
   loops, at least 25 independent checkable revisit pairs across multiple
   portions of the scan, all >=90% within the existing 1.5 nominal-mm tolerance.
   Report every omitted/uncheckable sample. Enough raw overlap must exist for
   those checks; otherwise the run is inconclusive, never an automatic pass.
   Checks used to fit loop constraints cannot also be the validation set.
   Existing good controls must not lose >1 percentage point at their existing
   checked pairs or develop a new tracking stop. Adverse cases must improve
   geometry or explicitly refuse uncertainty, not merely accept more frames.
3. **Known-truth and failure tests.** Analytic rigid transforms, reversal and
   pauses, dropped/duplicate frames, weak planes, wrong-overlap/symmetric
   hypotheses, occlusion and nonrigid intruders must exercise both success and
   refusal. Invalid/nonfinite inputs cannot become valid placements. Synthetic
   success does not qualify physical motion during structured-light exposure.
4. **Depth/coverage preserved.** Keep 20 ms as default. Faster modes remain
   separate experiments and must pass the earlier fixed-scene raw retention
   criteria, including edges, before promotion. Never remove a troublesome
   view, support fringe or feature after seeing its error to obtain a pass.
   Report whole-frame evidence and intended object coverage separately. CAD
   accuracy requires a confirmed matching target, existing frozen tolerances,
   placement-sensitivity checks and coverage of each intended test surface.
5. **Live service quality.** At baseline cadence, process all intended live
   frames with bounded memory/backlog. Target steady-state mean preparation +
   tracking <=180 ms and p95 <=220 ms on this PC; report recovery separately,
   its capture consequences and worst case. No hidden queue growth or confusing
   a coalesced preview with lost scan frames. Stop/loss/recovery UI stays usable;
   saved sessions reopen with identical validity. GPU failure is explicit.
6. **Repeatability and endurance.** After offline/integration gates pass, five
   fresh successful fixed-camera loops and five fresh handheld loops in the
   declared envelope, including reverse direction, changes in tilt, pauses and
   a return to the start. Different runs are retained even when they fail.
   This is an engineering acceptance sample, not a statistical guarantee of a
   population success rate. Then 15- and 30-minute continuous baseline runs
   pass capture/thermal/geometry checks; compare early/middle/late raw depth at
   like views. Flag CPU/NPU 75 C, stop 80 C; these are not certified optics
   limits. Temperature alone cannot qualify depth or dimensional accuracy.

## Execution map

| Stage | Work / decision | Evidence needed to advance | Can run without operator? |
| --- | --- | --- | --- |
| 0. Freeze and reproduce | Snapshot source/corpus, baseline core tests, reproduce the 1,171-frame case | Same decisions/poses/geometry; honest baseline failures retained | Yes |
| 1. Locate drift | Audit actual raw revisits, all saved candidate poses/references, local increments and preparation | Identify first/strongest divergence and distinguish measurement failure from selection/model bias | Yes |
| 2. One evidence-led correction | Unit-tested bug fix if proved; otherwise one closed local-registration or independently validated loop-correction experiment | Predeclare algorithm, fixed settings, fit/validation split and rejection rule before running it | Yes |
| 3. Regression and adversarial tests | New 20 ms controls, faster development case, older long adverse motion and two handheld recordings; known-truth fixtures | No new silent distortion, no good-control regression, improvement in the targeted failure | Yes |
| 4. Live integration | Integrate only a surviving change, preserve raw/pose provenance and no-live-mesh contract, serial performance and GUI lifecycle tests | Correct saved/reopened validity, bounded resources, measured latency and cancellation | Mostly; physical view must be suitable |
| 5. Fresh operating envelope | Freeze implementation; acquire new fixed and handheld validation runs; change view/direction, include loop returns | Fresh independent geometry checks and five repeats per intended mode | Handheld motion/target changes require operator |
| 6. Endurance / coverage | 15 then 30 minutes at the winning depth-preserving timing, replay representative and worst intervals; appropriate CAD check | No transport or thermal fault, no time-dependent raw/geometry collapse; coverage honest | Stationary hardware work yes; target setup may require operator |
| 7. Release decision | Re-run regression suite, document supported range/motion/surfaces and failure boundaries | Every applicable gate passed; no missing physical evidence presented as passed | Report only after gates actually pass |

Stage 2 is a decision based on stage 1, not a parameter lottery. A selection
experiment is justified only if the recorded alternatives contain better
placements; otherwise changing preference cannot repair shared bias. Local
registration work must be checked at known/raw-measured relative transforms.
Loop correction must use independent raw-cloud constraints, ambiguity and
overlap checks, and held-out image pairs; it must not fit a circular trajectory
and use the enforced return as evidence that handheld tracking is fixed.

Reject a specific experiment when it breaks an existing good control or only
moves errors into unchecked views. Preserve that failure, diagnose its mechanism,
and declare a genuinely different next hypothesis before testing it. Do not
repeat the failed keyframe-budget/preference/untrimmed/point-to-point or raw-
health archival-exclusion sweeps under new names. No threshold relaxation to
make a selected recording pass. Small regression fixes may be retained even
when the broader reliability gate remains unmet.

## Autonomous execution and boundaries

Execute stages 0–3 immediately, then proceed as far as evidence and existing
hardware allow. Use saved raw recordings first; do not ask for another handheld
attempt merely to reproduce the existing drift. Physical scanner use is
authorized, but never stream while another application owns it. Verify current
scene/motion from new observations before interpreting a physical comparison;
do not assume the last table state or object placement persists.

Work directly in this repository; preserve unrelated changes. No Coordinator,
external handoff, commit/push/deploy, dependency/driver installation or abandoned
live meshing. No firmware flashing: no authenticated image, calibration backup
and verified recovery route currently exist, and software registration is the
demonstrated unresolved problem. A concrete need for new authority or physical
operator action is reported as such, not silently assumed.

The original execution ledger is `docs/reliability-roadmap-results.md`; its
sealed `/home/ben/scans/diagnostics/20260908-reliability-roadmap/` archive is
historical and must not receive new files. New rounds have separate evidence
roots mapped by `docs/reliability-learning-log.md`, leaving all earlier sealed
roots unchanged. Label each stage completed, rejected, running
or awaiting physical evidence. Do not claim the destination merely because
the autonomous portion ends.
