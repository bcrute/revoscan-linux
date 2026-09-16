# M25: reverse initialization and agreeing-candidate preference

Declared 2026-09-09 04:56 UTC after [M21 Stage B](reliability-reverse-seed-stage-b.md)
fails, before implementation or M25 replays. Evidence:
`/home/ben/scans/diagnostics/20260909-reliability-seed-selection/`.
Current status: **complete and rejected**. All twelve observation controls
reproduce exactly. Forty changed trajectories run from 2026-09-09 06:30:52
until approximately 07:15 UTC; every original input is accepted. The complete
phase audit verifies all forty cells and all 840 required raw pairs. No whole
configuration survives: all eight preserve both stationary cases but fail all
three moving cases. All 420 seed/preference and 420 capacity/preference
contrasts are computable; no failed or unavailable pair is replaced.
The 07:20:27 closeout verifies all 170 executing files still match their frozen
and live copies. It counts 60,716 completed fit requests, two refused requests,
91,285 completed stages, 286,142 iterations and 63 unconverged reverse seeds.
The root is sealed at 07:22:32: 761 files / 1,953,033,916 bytes, manifest
`4af8bbcae42d5ff473b861f203d1461afde520e5d325c572ca844aaa746fe69d`.
An outside-root check under M26 verifies the exact inventory and all hashes
at 07:23:54. No further outputs belong in this sealed root.

The motivation is specific: every one of M21's 32 failing fixed-local checks
has a saved accepted adjacent candidate preserving its original pair guard.
Most bad long revisits have no good alternative left. This suggests testing
earlier selection history, not an oracle that picks a pose using future checks.

The diagnostic crosses forward/reverse initialization with model-first versus
adjacent-first preference, both 900/1800 capacities and original/centered finite
updates. Only preference **inside the original agreed candidate cluster** changes.
All candidate construction, ambiguity/refusal/rescue, overlap/travel/residual
gates, model1800, ten-neighbor normals and original raw inputs remain. Selected
overlap still affects original keyframe admission; that feedback is not erased.

Twelve original-preference controls must exactly reproduce the corresponding
M21 full inputs, decisions, poses, raw checks, solver and seed records. Forty
changed cells then preserve all 840 fixed pair records, with 420 seed-by-preference
and 420 capacity-by-preference contrasts. M22–M24 and unresolved survivors have
priority. M22–M24 have completed without survivors; the M21 Stage B rejection
is resolved. The executed observation
driver binds complete prior dispositions, all twelve original reports/scores/
seed traces and its own source before execution. No numerical GPU kernel is
changed by the preference experiment; exact recorded-GPU controls are mandatory.

The 76 focused tests exercise actual agreement arbitration, tied-cluster
refusals, model-only rescue, immutable policy/function identity, real numerical
tracker composition, both original adapters, recorded preference and report
hashes, partial failure preservation, damaged metadata, explicit CLI choices,
nonnesting and existing-output protection. Static reports historically lack a
preference field; the new companion records actual entry/exit preference rather
than rewriting those reports or claiming the field existed.

Executing source freezes at 05:09:48 UTC in 170 files / 4,498,760 bytes;
253 verification files / 6,771,019 bytes freeze separately. No production,
scanner, firmware, calibration, dependency, driver or mesh change. The selected
suite is not the full repository suite, and arithmetic tests are not accuracy,
handheld usability, live throughput or thermal qualification.

An initial narrow typing command omitted local source scope and produced 77
untyped-import/subclass errors. Explicit repository source checks then passed
80 files; the full selected command uses `MYPYPATH=src:tools` and passes 127.
No type errors were suppressed and no production code changed for that correction.

## Results

Minimum fixed raw-pair agreement below is percent, not physical object coverage.
The original requirement remains 90% for every pair and the separate good-pair
regression guard. Both stationary cases pass in every row; none of the moving
cases does. "Original" names the finite-update rule, not a production promotion.

| Capacity | Seed | Update | A1 | A2 | C |
| --- | --- | --- | ---: | ---: | ---: |
| 900 | Forward | Original | 1.840491 | 2.105263 | 2.276295 |
| 900 | Forward | Centered | 2.258727 | 2.067381 | 2.490272 |
| 900 | Reverse-forward | Original | 0.540054 | 3.133159 | 1.763668 |
| 900 | Reverse-forward | Centered | 0.271985 | 3.048780 | 2.758621 |
| 1800 | Forward | Original | 4.351768 | 2.017067 | 5.726141 |
| 1800 | Forward | Centered | 4.000000 | 2.091402 | 5.753647 |
| 1800 | Reverse-forward | Original | 1.453224 | 2.122016 | 6.060606 |
| 1800 | Reverse-forward | Centered | 1.621622 | 1.802575 | 6.374502 |

The motivating saved-candidate audit is not contradicted: an adjacent candidate
can improve one fixed local comparison at the old history, while selecting it
throughout a new history accumulates drift. The endpoint oracle was never a
guarantee about the changed trajectory. These results specifically reject the
declared adjacent-first combinations, not every possible combined method.

Across the forty runs, 20,192 offered inputs are accepted: 40 anchors,
20,105 adjacent selections and 47 original coarse-recovery selections. Model
and adjacent belong to the recorded agreeing set 17,642 times, with zero model
selections when adjacent agrees. Thus the actual selections corroborate the
declared preference, not just a metadata label. These are repeated algorithm
occurrences across reused recordings, not independent measurements.
The two refused fit requests are singularity failures in ordinary/reverse/
original A1 and dense/reverse/original A2; neither refuses a frame. One request
completed its reverse stage before the forward refusal, and that stage's 33
iterations are included in the completed-stage/iteration totals.

The conditional longer/handheld adapter is tested and separately sealed under
`20260909-reliability-seed-selection-stage-b` (437 files, 07:13:59 UTC). It has
63 focused / 1,370 selected tests and verified source snapshots, but zero recorded
qualification runs and no bound candidate. It must not be counted as Stage B
scanner evidence. No new Stage B execution is earned by M25.
