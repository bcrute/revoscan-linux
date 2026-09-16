# M22: larger ordinary model reference

Declared 2026-09-09 UTC while M19 changed trajectories were running, before
M20–M22 recorded-scan pilots. Evidence and the binding protocol are under
`/home/ben/scans/diagnostics/20260909-reliability-model-capacity/`.

Hypothesis: the 1,800-point model budget can discard useful reference samples
even when incoming/retained clouds are dense. Unlike M11, this test preserves
ordinary per-frame sampling and doubles the model budget to 3,600. It keeps
incoming/retained capacities 900/1800, both finite updates, eight reference
frames, original recovery budget, ten-neighbor normals and every quality gate.

The old 2,048-point normal-cache eligibility also controls GPU routing. The
changed diagnostic context raises eligibility to 4,096, with 32 entries and
actual target size at most 3,600. Per-fit provider records distinguish provider
use/cache hits from fresh GPU dispatch. Production source is untouched.

Completed 2026-09-09 05:15 UTC: six exact observation controls and twenty changed
trajectories; no whole survivor. All inputs are accepted and all 420 original
fixed pairs remain checkable. Both dense variants preserve four cases but fail
C; ordinary capacity preserves only the two stationary cases. All 61
focused tests pass; the selected M1–M22 suite passes 1,058 tests, lint and cold
typing of 119 files. The executing snapshot has 164 Python/SDK JSON files;
the separate verification snapshot has 248 files, frozen at 02:46:47 UTC.
A real numerical fixture above 2,048 points bypasses the provider under the
old route and uses it under the scoped route with an identical full fit result
for both finite updates. That provider is the CPU reference in unit tests;
real large-field CPU/GPU parity is recorded separately below.

The closed parity schedule reconstructs both model budgets at all 384 fixed
queries from F1/M1/M7's four grid controls, preserving missing models and saved
poses. Six exact observation pilots precede twenty changed trajectories.
M20/M21 completion and survivor priority remain binding. This is not a live
speed, scanner thermal, handheld or independently calibrated accuracy claim.

At the earlier 04:45 checkpoint, trajectories remained deferred for M21 Stage B. An additive
`AFTER-STAGE-B-REJECTION.md` and `m22/run-after-stage-b-rejection.py` recognize
only an exactly audited, complete rejection of M21's actual survivor. The new
plan will bind that audit/execution/plan and both driver versions. Original
protocol, driver, source, roster, settings and thresholds are untouched. All
17 eligibility-helper tests pass, including missing, partial and still-surviving
evidence. This amendment does not itself run or qualify M22.

## Complete trajectory result

Weakest required raw-pair agreement (%), not coverage or physical accuracy:

| Capacity / update | Static A | Static B | A1 | A2 | C | Cases passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 / original | 97.370 | 99.780 | 0.000 | 18.175 | 17.371 | 2/5 |
| 900 / centered | 97.370 | 99.780 | 0.000 | 12.552 | 26.821 | 2/5 |
| 1800 / original | 99.238 | 99.780 | 97.919 | 99.463 | 31.184 | 4/5 |
| 1800 / centered | 99.238 | 99.780 | 96.424 | 99.260 | 30.808 | 4/5 |

All six pilots reproduce original inputs, decisions, poses, raw checks and full
solver traces exactly. Dense A1/A2 also preserve the good-pair regression guard;
merely exceeding 90% is not the criterion. Both dense variants still fail C by
a large margin. The 210 density-by-budget contrasts and 420 original-budget
comparisons are all retained and computable. No model budget is promoted.

The cumulative audit through M22 records 575 changed trajectories, 115 whole
configurations, 12,075 required/checkable pairs and 98 exact observation pilots.
The sole Stage A survivor is the earlier M21 candidate, already rejected by its
complete Stage B; M22 adds none. This aggregate excludes later prepared phases.

All 30,148 fit requests complete, with 69,710 iterations and no fit refusals.
The normal provider is invoked once per request; 7,349 fields exceed the old
2,048-point routing limit, with a maximum of 2,876 points. Every recorded field
preserves point identity and ten-neighbor support. Provider requests may hit
the exact cache: these counts are not all fresh GPU dispatches. The separate
728-field parity test does record 728 actual GPU eigensolve calls. All 164
executing source files remain byte-identical to saved and live source.

## Large-field parity completed, 2026-09-09 03:04 UTC

All 384 prescribed historical model queries are retained, including twenty
without a formed model. At both budgets on the other 364, all **728 complete
NormalFields / 1,137,627 estimates** match the CPU reference exactly. There are
190 fields above the original 2,048-point routing limit; the maximum has 2,734
points. The RX 9070 XT executes exactly 728 eigensolve dispatches. The audit
takes 72.632 seconds with no other GPU experiment running, though unrelated CPU
verification is concurrent; this is not a live or isolated throughput benchmark.

Each original 1,800-budget model count/voxel count reproduces its saved candidate
and prior utilization audit. The 3,600-budget field uses the same saved poses
and contributors, not newly fitted poses. No query substitution, model fitting,
CPU fallback or physical scanner access occurs. This demonstrates tested
larger-field arithmetic support, not better tracking. At that parity checkpoint,
M22 trajectory pilots remained behind M20/M21 and survivor Stage B priority;
the completed trajectory result is recorded at the top of this document.

The additive descriptive count audit retains every paired query in
`large-model-budget-effect.json`. All eighty ordinary stationary formed-model
point hashes are unchanged by the larger allowance. Dense static B median
voxel counts rise from 1,227 to 1,602.5 / 1,601.5 for original/centered saved
histories. Moving per-configuration median paired ratios are about 1.68–1.74.
More points also change the pooled normals' spatial support at fixed ten
neighbors; this is not an isolated normal-support experiment. Wrong saved
poses can inflate occupied voxels, so these counts do not measure accuracy.

The completed archive is sealed at 2026-09-09 05:20:14 UTC: 581 files,
670,960,732 bytes, manifest SHA-256
`494456b6bb44cee0b2f523ee7ae3f595f13e985f4fa7b9348e26e35519206167`.
An outside-root inventory/hash verification passes at 05:22:37 UTC and is
retained in the M23 evidence root. This is local integrity, not an off-machine
backup or proof of measurement accuracy.
