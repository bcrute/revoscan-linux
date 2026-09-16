# Voxel-first model plus normal support

Continuation of [the mechanism campaign](reliability-mechanisms.md).
Evidence: `/home/ben/scans/diagnostics/20260909-reliability-voxel-normal-combination/`.
Completed: six exact observation controls, twenty changed trajectories, no
whole-configuration survivor. All 420 required pairs and interaction contrasts
are checkable. Every input is accepted, including the badly misplaced runs.

M11's dense voxel-first model preserves both stationary controls and A1/A2, but
fails C. M13's dense twenty-neighbor normals preserve all moving controls and
static A, but regress on static B. M17 tests their combination directly using
the unchanged component implementations, both 900/1800 capacities and both
original/centered updates. It does not infer that complementary partial results
must combine successfully.

Six exact M11 observation pilots precede twenty changed five-case trajectories.
All 420 voxel-model-by-normal-support pair contrasts are retained, with missing
values never imputed. All original full-acceptance, 90%-agreement and one-point
no-regression guards remain, followed by the longer/handheld, fresh physical,
coverage, GUI/service and thermal gates. No case-specific switching, new crop,
calibration, normal estimator, policy threshold or experimental mesh.

The execution snapshot contains 154 Python/SDK JSON files. A separate 243-file
verification-source snapshot preserves the test/config source. Thirty-eight
focused tests pass, including exact real-tracker, model-record and solver-record
composition against explicit normal counts on a three-plane fixture. That
fixture uses a CPU reference provider; actual GPU behavior remains subject to
the existing 232-cloud parity record and the mandatory real-replay pilots.
No production file, scanner setting, firmware or driver is changed.

All four configurations preserve both stationary controls. At ordinary density,
both updates fail every moving case: A1 reaches 0% minimum agreement, A2
13.8739% / 11.5891%, and C 6.4849% / 3.3948% (original / centered). Dense original
preserves A1 (98.5897%) but fails A2 (41.5995%) and C (4.9603%). Dense centered
preserves A2 (99.5460%) but fails A1 (0%) and C (7.0234%). Thus neither dense
combination retains all the moving benefits of normal support alone. These are
whole-configuration rejections, not a rationale for switching methods per case.

The aggregate through M17 verifies 435 changed trajectories, 87 whole
configurations, 9,135 fixed pair checks, 68 exact observation pilots and zero
survivors. The distinct [coarse-to-fine experiment](reliability-multiscale.md)
continues next. Reusing these recordings is development evidence, not blind
qualification.

The combined selected suite passes 697 tests, lint and cold typing across 109
source files. These are implementation checks, not scanner qualification.

The launch checks require unchanged M11/M13 source dependencies and completed
M14/M15/M16 audits with no whole survivor. M16 executes in its separate
`20260909-reliability-covariance-descent` archive after a location-only copy of
its unchanged method/source. Any survivor takes Stage B priority.
At most two GPU children run at once, with the original 900/960-second bounds.

A separate read-only alternating-delivery diagnostic checks all 640 stationary
inputs on each capture's all-frame persistent valid pixels (52,052 / 61,326).
Median absolute odd-minus-even mean-depth offsets are 0.00516 / 0.00541 nominal
mm, with 99th percentiles 0.06844 / 0.06137. The fixed-half maps correlate
0.342 / 0.554, but cross-capture correlation on 36,941 common pixels is only
0.0174. Host-delivery parity is not verified DSP phase. These results do not
justify a general spatial correction; no depth correction, pose fit or tracking
crop is applied. Full maps and input hashes remain in the diagnostic archive.
