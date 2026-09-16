# Reliability continuation: combined-factor test

8 September 2026. The user authorized testing combinations after the
[standalone model-normal experiment](reliability-reference-model.md).
**Completed: all sixty Stage A cells were executed and scored. None of the
twelve configurations survived all five recordings; no combination is promoted.
Stage B was not run, as required by the predeclared survivor gate.**

Evidence root: `/home/ben/scans/diagnostics/20260908-reliability-factorial/`.
Its predeclared `PROTOCOL.md` and `stage-a-plan.json` preserve the fixed factors,
input hashes, original pair scores and advancement rules.

## Frozen design

| Factor | Unchanged implementations combined in this round |
| --- | --- |
| Sampling | Production compacted stride; S2 fixed pixel rank; S3 hierarchical grid |
| Ordinary model normals | Pooled-target PCA; R2 native-frame PCA at unchanged voxel centroids |
| Convergence | Residual change; S1 residual plus surface movement |

Every one of the twelve configurations receives all five first-stage recordings:
static A/B with all 320 inputs and moving A1/A2/C with the original 628 settled
inputs. All sixty cells are attempted, regardless of other cells' failures.
The original refusal limit remains active. No thresholds, budgets, component
implementations, reference policy or raw inputs are retuned. Actual GPU normals
are mandatory; SciPy correspondence search remains unchanged.

Six pilot cells reproduce archived full decisions exactly: baseline plus the
four single-change modes on A1 and baseline static A. Poses, moving coherence,
static apparent movement and all original static checks match. The pilots are
six of the sixty cells, not independent new captures. A separate pilot scoring
check reproduces the archived standalone verdicts.

Fixed scoring retains every original pair: twenty static checks per view,
13/14 moving A1/A2 pairs, and C's fifteen original plus 23 credible common
checks. The common C schedule retains all forty raw anchors and has only 22
endpoint-separated checks. It cannot independently qualify a scan.

Each complete configuration must pass all five cells: every required input
accepted, all required checks projectable and >=90% within 1.5 nominal mm,
and no >1-percentage-point original-pair regression on the good controls.
Static A is adverse and must clear all twenty 90% checks. Missing poses,
unoffered suffixes, exceptions and timeouts remain explicit failures/omissions.

Paired main effects and two-/three-factor interactions retain their constituent
scores. Missing scores are never imputed as zero. These are descriptive results
on reused data, not statistical significance or independent release evidence.
Only complete first-stage survivors advance to D, the extended development
case, both older adverse long recordings and both archived handheld failures.
Fresh operating-envelope, GUI, known-truth/coverage and endurance gates remain.

No production source, scanner settings, firmware, drivers, dependencies or live
mesh changes are made. This is offline replay, with at most two isolated GPU
processes and the existing time bounds. No new scanner-temperature measurement
is claimed. Earlier source/evidence records are preserved.

## Complete Stage A result

The table gives the weakest agreement (%) on each cell's **fixed required raw
pairs**, within the existing 1.5 nominal-mm tolerance. These are geometric
agreement fractions, not frame acceptance, object coverage or calibrated
accuracy. A minimum above 90% is necessary but not sufficient: original-pair
regression, missing evidence and input-acceptance guards still apply.

| Sampling / normals / convergence | Static A | Static B | A1 | A2 | C |
| --- | ---: | ---: | ---: | ---: | ---: |
| stride / pooled / residual (baseline) | 69.00 | 93.84 | 96.50 | 98.48 | 95.39 |
| stride / pooled / settling | 44.00* | 94.74 | 8.92 | missing* | 91.02 |
| stride / native / residual | 70.41 | 95.97 | 96.57 | 97.37 | 0.07 |
| stride / native / settling | 38.82 | 95.10 | 93.76 | 96.51 | 86.57 |
| rank / pooled / residual | 97.50 | 99.78 | 96.96 | 13.72 | 21.81 |
| rank / pooled / settling | 97.56 | 99.78 | 97.00 | 95.03 | 16.10 |
| rank / native / residual | 99.24 | 99.78 | 94.63 | 40.45 | 15.93 |
| rank / native / settling | 99.24 | 99.78 | 30.14 | 8.19 | 17.03 |
| grid / pooled / residual | 97.37 | 99.78 | 98.57 | 97.87 | 17.08 |
| grid / pooled / settling | 97.43 | 99.78 | 99.28 | 97.96 | 12.12 |
| grid / native / residual | 97.37 | 99.78 | 98.65 | 42.82 | 2.95 |
| grid / native / settling | 97.43 | 99.78 | 15.45 | 35.13 | 16.42 |

\* Stride/pooled/settling has two missing static-A checks (303/320 inputs
accepted), so 44.00% is only the minimum of its eighteen checkable pairs. On A2
it stops after 366 offered inputs, accepts 362/628, and leaves all fourteen
required pairs uncheckable. These omissions are failures, not zero scores.

All eight rank/grid configurations pass both static cells, but none clears
the moving controls. In particular, C rejects every nonbaseline configuration.
Even stride/pooled/settling, with a C minimum above 90%, loses 8.625 percentage
points on an original pair and accepts only 626/628 inputs. The baseline passes
the other four Stage A cells but fails static A; its earlier long/handheld
failures also remain. This is not simply twelve failures on the one-point guard.

Across repeated replays there are 30,288 input slots, 30,026 offered inputs,
29,972 accepted inputs, 54 refusals and 262 unoffered suffix inputs. All sixty
processes exit successfully with completed reports and verified actual GPU
normals; scientific tracking failures remain in those reports. Of 1,260 required
pair records, 1,244 are checkable, 517 of those are below 90%, and sixteen are
missing. These totals reuse five recordings and include overlapping schedules;
they are not independent physical captures or unique observations.

## What the combinations actually change

The user's combination hypothesis was warranted. Individual failures did not
predict the combined trajectory, and some conditional improvements are large:

- On A2's same frozen raw pair **113 / 625**, rank sampling alone scores
  **13.7161%**; rank plus settling scores **95.0327%**, an improvement of
  **81.3166 percentage points**. Adding settling to rank improves all fourteen
  A2 pairs by 73.0473–85.4628 points. Nevertheless that combined A2 replay
  refuses one input and loses up to 4.3352 points against the production baseline;
  the same configuration reaches only 16.10% on C. The full two-factor contrast
  against baseline and settling alone is unavailable on A2 because the latter
  stops before the required late endpoints. The conditional effect is not
  substituted for a complete interaction estimate.
- A fully checkable two-factor interaction occurs on C's original pair
  **17 / 520**. Baseline, settling alone, native normals alone, and native plus
  settling score respectively **99.8921%, 99.8380%, 0.0820%, 99.8921%**. The
  difference of differences is **+99.8643 percentage points**. This is a genuine
  matched-pair interaction, but the combined configuration still has another
  required C pair at 86.57%, as well as failures on other recordings.
- Rank plus native normals reduces static A's maximum apparent surface motion
  to **0.0757 nominal mm**, with all twenty agreement checks at least 99.2376%.
  Adding settling gives 0.1024 nominal mm and the same weakest agreement. Both
  accept all 320 static inputs. These stationary diagnostics do not establish
  physical pose accuracy or handheld tracking.
- Combining all three changes with rank accepts **every one of the 628 inputs
  on both A1 and A2**, yet the weakest fixed geometry checks are **30.14% and
  8.19%**. Continuous acceptance is emphatically not evidence of a correct scan.
- Grid plus settling reaches a 99.28% A1 minimum and a maximum original-pair
  loss of only 0.1196 points, but refuses source 215. Its A2 original-pair loss
  is 1.4126 points and its C minimum is 12.12%. Selecting its attractive A1
  minimum would conceal the failures elsewhere.

The audit retains 33 contrast definitions per pair: twenty conditional main
effects, eleven two-factor interactions and two three-factor interactions.
Of 3,465 contrasts, 3,289 are computable and 176 retain missing constituent
scores. All constituent scores and coefficients are recorded. These are
descriptive comparisons on reused recordings, without significance or
population-level claims. No post-hoc choice of a different configuration for
each recording is evaluated or validated here.

## Verification, scope and handoff

The final selected software suite has **284 passed, one skipped, no failures**;
the skipped opt-in real-GPU reference comparison then passes in a separate
hardware-enabled run. The earlier 70-test factor/context and 67-test
audit/context suites also pass, but overlap these tests and are not added to
the unique test count. Ruff, targeted mypy (82 source files), and whitespace
checks pass. This is not a new run of the entire repository test suite.

The runner freezes all twelve independent factor selections, preserves complete
input manifests and failure records, and prevents overlapping process-local
patches. The scorer validates metadata, actual GPU activation, accepted poses,
native-model call provenance and unchanged report hashes before scoring the
original schedules. Its outer 90% promotion gate does not use the embedded
legacy projection report's 75% `coherent` flag as a promotion verdict.

All 79 production `src/scanner3d` files are byte-identical to this round's
starting snapshot. The only altered pre-existing source/test/tool file is
`tools/scanner_motion_replay.py`, which adds an explicit offline factor API;
five new configuration/runner/audit/test files cover the experiment. The
existing default trajectory reproduces exactly. All six prior evidence roots
retain their exact inventories and hashes. The first preservation-report
adapter failed while adding a duplicate dictionary key; its failed report is
retained, and a corrected adapter verifies every root and source snapshot.
This reporting-only correction changes no replay or scoring result.

Two retained host-sensor snapshots show the PCI 03:00 GPU at edge 46–47 C,
junction 48–49 C and memory 71 C; CPU package readings are 73–75 C. No alarm is
reported in those snapshots. They are neither continuous peak measurements nor
scanner-chip/optics temperatures. No scanner was opened by this experiment and
no thermal-endurance claim follows. Concurrent replay timings are not isolated
GPU throughput or live-GUI FPS measurements.

The complete raw-pair audit is `stage-a-audit.json`; `execution-summary.json`
provides the compact matrix, actual counts, verification and advancement
decision. `PROTOCOL.md`, frozen plan, six pilot verifications, sixty complete
cell reports/logs, software tests, original/executed/scoring/final source
snapshots and the final integrity manifest retain the evidence trail.

**Decision:** this specific twelve-configuration family is exhausted at Stage A.
Long-recording/handheld Stage B, app integration and fresh endurance validation
are not executed because no candidate meets the unchanged entry gate. This
rejects the tested configurations, not the general possibility of combining
methods. The remaining problem is view-dependent wrong pose acceptance and
its propagation through reference/model history; the tests do not yet prove
which internal step causes it. A subsequent correction needs a new, separately
declared causal hypothesis and regression experiment, not an unvalidated
per-recording switch between the best-looking rows. The overall reliability
roadmap remains incomplete.
