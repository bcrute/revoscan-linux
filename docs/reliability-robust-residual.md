# Residual-based robust weighting

M20 continues [the reliability roadmap](reliability-roadmap.md).
Evidence: `/home/ben/scans/diagnostics/20260909-reliability-robust-residual/`.
Completed 2026-09-09 03:40 UTC: all forty changed trajectories accept every
original input, but none of eight whole configurations survives. All 840
required pairs are checkable. Six observation pilots reproduce full original
inputs, decisions, poses, raw scores and solver traces exactly.

Earlier weighting emphasized source planarity and normal-direction balance.
This experiment instead reduces the influence of large current target-plane
residuals, following the weighted-equation approach described in
[Open3D's robust-kernel tutorial](https://open3d.org/docs/latest/tutorial/t_pipelines/t_robust_kernel.html).
It is an independent implementation using the existing solver, not an Open3D
dependency or performance claim. Huber and Cauchy use one predeclared scale:
the larger of one original raw depth code and median absolute residual divided
by 0.6744897501960817. This zero-centered absolute statistic is sign-invariant;
it is not median-centered MAD or a measured scanner-noise bound.

Both kernels cross ordinary/dense source capacity and original/centered finite
updates. Existing correspondences, distance trim, model, normal estimator,
unweighted residual reporting, stopping, singularity and acceptance guards
remain. Observation delegates to the actual original solver. Changed steps
retain scale, weight/effective-count summaries, actual delta, and frozen-scale
robust costs before/after; a lower robust cost cannot replace a raw-quality gate.
There is no line search, pose hold or silent fallback.

Six exact F1/M7 observation pilots precede forty changed trajectories, with
M13's exact controls supplying full solver traces. Preserve all 840 required
raw comparisons, paired L2/Huber/Cauchy differences and 420 density-by-kernel
contrasts. M18/M19 must complete first; any whole survivor has Stage B priority.
Reused recordings are development evidence, not independent qualification.

All 86 focused tests pass, including independent weighted equations, known
transforms, deliberate outliers, arbitrary signs, singular refusals, full tracker
composition, actual observation-factory identity and report/scope preservation.
Executing source is frozen in 160 Python/SDK JSON files; 246 verification-source
files are separate. The selected M1–M20 suite passes 932 tests, lint and cold
typing of 115 files. No production, firmware, dependency, driver, scanner,
calibration, crop or experimental-mesh change.

## Complete result

Weakest required raw-pair agreement (%), not coverage or calibrated accuracy:

| Capacity / kernel / update | Static A | Static B | A1 | A2 | C | Cases passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 / Huber / original | 97.569 | 99.780 | 10.875 | 18.813 | 13.909 | 2/5 |
| 900 / Huber / centered | 97.686 | 99.780 | 95.761 | 13.806 | 14.733 | 2/5 |
| 900 / Cauchy / original | 97.441 | 99.780 | 4.819 | 11.577 | 13.366 | 2/5 |
| 900 / Cauchy / centered | 97.748 | 99.780 | 5.591 | 8.583 | 15.645 | 2/5 |
| 1800 / Huber / original | 99.238 | 98.943 | 99.156 | 97.381 | 28.123 | 3/5 |
| 1800 / Huber / centered | 99.238 | 98.943 | 96.781 | 93.130 | 21.443 | 2/5 |
| 1800 / Cauchy / original | 99.238 | 99.779 | 97.314 | 14.786 | 27.214 | 2/5 |
| 1800 / Cauchy / centered | 99.238 | 99.779 | 96.660 | 18.341 | 26.612 | 2/5 |

Every configuration preserves both static controls. Dense original Huber also
preserves A1, but A2 loses 1.986 percentage points on an original pair and C
fails outright. The other above-90 moving minima still fail the unchanged
one-point regression guard. No configuration preserves C. Downweighting current
residuals is not sufficient to preserve this changing reference history.

The audit retains all 420 original/Huber/Cauchy comparison rows and all 420
fully computable density-by-kernel contrasts. The aggregate through M20 checks
535 changed trajectories, 107 whole configurations, 11,235 required pairs,
86 exact observation pilots and zero survivors. Nothing is promoted.

All 142,921 solver steps complete and strictly reduce their frozen-scale robust
cost; zero steps fail or use the raw-code scale floor. Recorded scales range
0.02075–4.19866 nominal coordinate units, and effective weighted-count fractions
range 0.63449–0.99802. The successful arithmetic descent therefore does not
explain away the failed geometry. Per-cell summaries and every step remain in
the archive; these are repeated solver occurrences, not independent points.

The immutable evidence archive contains 652 files / 1,667,760,239 bytes,
sealed at 03:45:09 UTC with manifest SHA-256
`c4a769062d38e943e6bffdf4fbb969caf7dfd666554c06cc6426bc9b645a52b6`.
The separate M21 root retains the exact outside-root inventory/hash verification.
