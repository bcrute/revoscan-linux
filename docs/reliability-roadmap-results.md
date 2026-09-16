# Reliability roadmap execution — 8 September 2026

Plan: [reliability-roadmap.md](reliability-roadmap.md).

This page records the first execution round. Subsequent physical captures,
registration experiments and diagnostic improvements are recorded in
[the continuation](reliability-registration-continuation.md).

Status: baseline/drift audit and the first closed offline correction/regression
round completed. **No correction survived the integration gate.** This is not
completion of the overall reliability roadmap. Original 20 ms timing and
production tracking are unchanged. The current physical table state has not
been assumed; no new hardware capture or firmware operation was performed.

Evidence root: `/home/ben/scans/diagnostics/20260908-reliability-roadmap/`.

## Baseline and drift diagnosis

The full 1,171-frame GPU replay reproduced every complete tracking decision,
pose, accepted source index, input record and coherence result exactly.
Core baseline regression: **856 passed, one skipped**, 531.251 seconds, across
geometry, compute, both registration solvers, preparation, tracking/recovery,
relocalization, both graph solvers, revisit/coherence and return-health tests.
These concurrent runs are not performance benchmarks.

The raw-revisit candidate audit preserves every saved alternative at each
existing checked pair, with original references and gates. At the worst
extended pair, adjacent/keyframe/model give **76.92% / 74.14% / 76.45%**;
switching preference cannot repair that recorded state. Other views do have
better alternatives, but a post-hoc alternative is not a causal replay of a
selection policy. Both new good 20 ms controls were audited too.

## Experiment L1: rejected

`LOOP-EXPERIMENT.md` predates the first fits. Fixed even/odd sample roles and
four-frame endpoint exclusion separate new closure constraints from validation.
Of 12 proposed closure fits, 11 pass descriptor, forward/reverse raw projection,
existing local ICP and cycle gates. The first is correctly rejected on cycle
error 1.2315 nominal mm against the unchanged 0.65 limit.

The existing 59-node / 69-edge SE(3) graph optimiser exhausted 100 iterations.
Its mixed tangent objective fell 139.438 → 0.442, yet its separately reported
worst geometric edge was 1.344 radians / 468.37 nominal mm. This is a failed
offline correction, not a diagnosis that this optimiser caused the original
live drift (it was not active there). No failed candidate poses were applied to
the app. The iteration limit, split and weights were not tuned around failure.

## Experiment L2: rejected

`SURFACE-EXPERIMENT.md` specifies a different graph objective: displacement of
the measured source-cloud points, in consistent depth units. Four
moment-matched landmarks exactly reproduce that cloud's quadratic displacement
metric; the method does not assume a turntable axis or constant velocity.
The raw links, split, node grid and weights remain L1's fixed choices.

Initial mathematical/guard tests check full-cloud metric equivalence,
sensor-coordinate invariance, known noncommuting poses, exact anchors,
interpolation of every pose, held-out endpoint protection, disconnected graphs,
nonfinite/coincident points, absent depth and a six-DOF-ambiguous plane.
The final combined selection has **14 passing tests**. The first surface-test
run retained one fixture error (a generator passed to a Sequence API), fixed
in the test without changing any geometric threshold. No physical reliability
claim follows from these analytic fixtures.

The fixed SciPy sparse solve exhausted 100 evaluations (1,601 actual residual
evaluations): physical objective 14.8587 -> 0.22895, worst surface edge RMS
0.81238 nominal mm. No held-out qualification was run after nonconvergence.

## Experiment L3: rejected for unconditional integration

`ANALYTIC-EXPERIMENT.md` retains L2's entire physical objective and graph but
uses analytic Jacobians and joint, diagonally damped LM updates. Independent
finite-difference Jacobian and known-truth tests pass (eight surface tests).
The development graph converges in ten trials. Its ten held-out pairs improve
from a minimum 76.45% to 93.99%; all 22 credible pairs exceed 92.25%. The worst
fixed-local change is -0.546 percentage points, within the one-point guard.

However, the good controls lose up to **2.14 / 1.45 percentage points** on
existing checked pairs. Their tool-level `candidate` status checks minimum
agreement, not the roadmap's additional good-control regression requirement;
the roadmap therefore rejects unconditional L3. No production change follows.

## Experiment G1: correction rejected; unchanged controls preserved

`GUARDED-EXPERIMENT.md` predeclares conditional correction based only on the
fixed fit set's original agreement. Already-consistent scans retain exact poses;
independent validation can reject that choice but cannot trigger a second try.
Correction uses unchanged L3 mathematics. Every credible check must exceed 90%,
and previously good checks cannot lose over one percentage point. Missing loops
remain insufficient. This is a separate workflow hypothesis, not a relabelled
unconditional L3 pass or a solution to live handheld tracking loss.

Both good controls take the no-op branch and preserve every pose exactly.
The development correction is rejected: one previously good held-out pair
falls 99.49% -> 93.99%, a 5.50-point loss. The old/later long-motion recordings
have held-out minima 35.58% / 75.63% and are rejected too. Both handheld cases
lack a credible automatic revisit set: insufficient evidence, no fitted poses.

## Experiment G2: denser constraints, still rejected

`DENSE-EXPERIMENT.md` predeclares raw-only proposals every 32 accepted frames,
minimum separation 100, with both endpoints excluded within four frames of
every credible held-out endpoint. One missing-aware best match per query;
duplicates, unsuitable matches and failed bidirectional fits remain reported.
The graph bound grows to 256 nodes for this measurement schedule; numerical
iterations, physical objective, weights, activation and rejection gates do not
change. This is a coverage hypothesis, not a numerical parameter sweep.

| Known recording | Admitted new raw links | Weakest held-out pair after correction | Result |
| --- | ---: | ---: | --- |
| Good 20 ms A1 | 0, not needed | 98.54% unchanged | Original poses preserved exactly |
| Good 20 ms A2 | 0, not needed | 98.57% unchanged | Original poses preserved exactly |
| Extended 14 ms, 1,171 frames | 16 | 96.06% | Rejected: another sampled pair is 88.17%; good-pair regression |
| Old adverse, 2,703 frames | 77 | 69.00% | Rejected: geometry and local/regression guards |
| Later adverse, 2,359 frames | 62 | 91.89% | Rejected: good-pair loss up to 3.01 points; local loss 1.020 points |
| Handheld move/pause, 188 inputs | None | Uncheckable | Insufficient loops; existing tracking loss is not repaired |
| Combined handheld/table, 303 inputs | None | Uncheckable | Insufficient loops; no handheld accuracy claim |

The good-control held-out percentages above use only the fixed validation
subset. Their complete original checked-pair minima remain 96.50% / 98.48%.
The complete paired results and exact indices are in `execution-summary.json`.
Do not select only the improved minimum: the extended case loses up to 11.01
points elsewhere, and the old adverse case loses up to 3.34 points locally.
All three corrected G2 outputs are rejected; no selected pose set is emitted.

These results establish that minimising a pose-graph objective and fitting raw
loop endpoints are insufficient to preserve the intervening geometry. They do
not establish that every possible loop-closure method fails, or that scanner
firmware is the cause. In particular, no circular trajectory or constant-speed
assumption was fitted, and none of these experiments fixes live local tracking.

## Verification, retained work and evidence integrity

Three disjoint final test selections total **1,188 passed, two skipped**:

- Core baseline: 856 passed; one existing synthetic-period/minimum-lag skip.
- GUI/capture lifecycle: 296 passed; one explicitly opt-in real-GPU test skipped.
  Covers scripted Stop, loss/recovery, pause, short frames, saved/reopened project,
  live point-model/no-mesh behavior, preview provenance, acceleration, asynchronous
  coverage and immutable buffer return. This is not a physical GUI FPS benchmark.
- New diagnostic tests: 36 passed. Exact cloud metric and analytic Jacobians,
  known-pose recovery, anchored interpolation, validation exclusions, empty/planar
  geometry refusal, raw-hash failure, preserved failed reports, no overwriting,
  conditional selection and stricter regression rejection are exercised.

Ruff, scoped mypy and `git diff --check` pass. The final summary also rechecks
every source-report hash and all offered/accepted index mappings, verifies raw
hash accounting and held-out endpoint exclusion, and asserts that rejected or
insufficient workflows have no selected pose output. Both good controls' no-op
outputs compare exactly to original source poses. Eleven snapshotted production
modules compare byte-for-byte with their baseline snapshots.

Intermediate failures are retained: the first surface test's fixture error;
the first lifecycle selection named a nonexistent test file; one mypy invocation
omitted the source search path and a later new list needed a type annotation.
Corrected invocations passed. These do not turn rejected geometry into a pass.

Retained repository changes for this execution are the roadmap/results and
offline candidate, loop, surface-metric, guarded-policy and dense-proposal
diagnostics with their tests. No GUI or production policy selects a new solver.
No experimental live mesh, firmware flash, dependency/driver installation,
commit, push, or modification of earlier sealed evidence roots.

## Where the roadmap stands, and the next engineering gate

| Stage | State after this execution |
| --- | --- |
| 0–1: freeze, reproduce, audit alternatives | Completed for this closed round; shared placement bias demonstrated |
| 2: measured correction | L1/L2 nonconvergent; L3/G1/G2 rejected for integration |
| 3: regression and failure checks | Seven-capture G1/G2 corpus and listed tests completed; full release qualification not passed |
| 4: live integration | Not entered: no geometry-preserving candidate |
| 5: fresh operating envelope | Pending a surviving implementation and operator handheld motion |
| 6: 15/30-minute endurance and dimensional coverage | Deferred by the earlier gates; no new temperature or accuracy claim |
| 7: reliable release | **Not qualified** |

The next engineering milestone is a correction that preserves trustworthy
local geometry between raw revisits, together with a local tracker that handles
the recorded handheld failures. Before another implementation, measure the
per-view relative-pose uncertainty and whether residual deformation originates
within a depth measurement or accumulates through reference updates. That
distinguishes a need for uncertainty-weighted/submap registration from a need
to refuse motion-corrupted depth. Sparse or denser loop fitting alone did not
answer it. Do not repeat this round by changing weights until a selected scan
passes, or ask for more handheld recordings merely to repeat known failures.

The integration prerequisite, not lack of permission to operate the scanner,
is why this execution stops before fresh/endurance qualification. The complete
roadmap remains open; failed earlier gates cannot be bypassed by a successful
thermal soak. Evidence, source snapshots, protocols, original failed runs and
machine-readable summary are retained under the stated root and covered by its
final `evidence-manifest.json` (file integrity, not measurement certification).

The primary manifest covers 243 files / 60,383,603 bytes. A final source-copy
directory collision left the complete diagnostic-dependency directory for a
separate fresh snapshot; it is preserved under
`/home/ben/scans/diagnostics/20260908-reliability-source-supplement/` with its own
manifest. The primary sealed root was not reopened or changed. The five new
diagnostics were already individually included in that primary snapshot.
