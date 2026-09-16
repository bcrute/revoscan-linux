# Coarse-to-fine registration test

M18 continues [the reliability roadmap](reliability-roadmap.md).
Evidence: `/home/ben/scans/diagnostics/20260909-reliability-multiscale/`.
Completed: six exact observation pilots, twenty changed trajectories, no whole
survivor. All 420 required pairs and 210 density interactions are computable;
the paired single-level comparisons number 420. All inputs are accepted.

| Configuration | Passing cases | Binding failures |
| --- | --- | --- |
| Ordinary, original update | 4/5 | C minimum 43.5243% |
| Ordinary, centered update | 4/5 | C minimum 28.9001% |
| Dense, original update | 4/5 | Static B loses 1.9122 percentage points |
| Dense, centered update | 2/5 | Static B / A1 / A2 lose 2.3095 / 1.0345 / 1.7815 points |

Dense original moving minima are A1 99.3007%, A2 98.6521%, C 98.8413%; static
A is 96.1886% and B 97.2958%. Dense centered gives A1 98.2211%, A2 97.5864%,
C 98.8386%, static A 95.8065%, B 97.2394%. High minima do not override the
unchanged original-pair regression guard. No per-case method switching follows.

The aggregate through M18 verifies 455 changed trajectories, 91 whole
configurations, 9,555 checkable fixed pairs, 74 exact pilots and zero survivors.
[Source-normal compatibility](reliability-normal-compatibility.md) is next.

Each original tracker fit runs the existing ICP at voxel sizes 6.0, 3.0 and
1.5 nominal units, using each transform to seed the next level. All levels
reduce the same original bounded source and target independently. The original
ten-neighbor normals, correspondence reach, trim, tolerance and fifty-iteration
limit remain per level. Only the actual final-resolution result reaches the
original tracker gates. Coarse refusal propagates; there is no single-level
fallback or coarse pose acceptance. Separate records expose all levels and
total completed iterations, with an explicit 150-iteration maximum per requested
fit. This is not an unchanged-compute or live-speed claim.

The algorithmic idea is described in [Open3D's own multi-scale ICP tutorial](https://www.open3d.org/docs/release/tutorial/t_pipelines/t_icp_registration.html#multi-scale-icp-example).
This experiment composes our existing solver, not Open3D code or its example's
parameters. It installs no new dependency.

The closed grid/pooled/residual roster crosses ordinary/dense preparation and
source/retained capacities with original/centered updates. Six exact F1/M7
observation pilots precede twenty changed five-case trajectories. All original
full-input, raw-geometry and no-regression checks remain, with 420 matching
single-resolution comparisons and 210 density-by-pyramid contrasts. No per-case
method selection, new crop, imputed pose, scanner/firmware or production change.
M16/M17 must finish first and any whole survivor takes Stage B priority.
Both prior phases are now rejected. A separate pre-matrix pilot audit completes
the protocol's full solver-trace/raw-check guard, which the original driver
did not explicitly enforce. All six pilots match every raw check and all
23,315 solver records from M13's exact ten-neighbor controls (F1 predates solver
tracing). The frozen driver, plan and pilot outputs were not rewritten.

The 156-file executing source and separate 244-file test/config snapshots are
frozen. Sixty-nine focused tests pass, including real numerical and full-tracker
composition, exact output object/argument identity, level order and seed chaining,
metadata, refusal propagation, bounds and scope restoration. The initial mock
fixture called a nonexistent transform convenience method; it was corrected
before numerical tests and source freeze, and the failed XML is retained.
All 576 GPU/reference normal-field comparisons pass exactly: the previous 192
fixed raw-grid query/capacity combinations at all three declared resolutions,
24.059 seconds, actual RX 9070 XT dispatches. This establishes numerical parity,
not physical normal truth, motion accuracy or throughput qualification.

The combined M1–M18 selected suite passes 766 tests, lint and cold typing across
111 source files. These are implementation checks, not scanner release gates.
