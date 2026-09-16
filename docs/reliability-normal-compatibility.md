# Source-normal eligibility and direction compatibility

M19 continues [the reliability roadmap](reliability-roadmap.md).
Evidence: `/home/ben/scans/diagnostics/20260909-reliability-normal-compatibility/`.
Six recorded-scan observation pilots exactly reproduce the original trajectories,
full raw checks and solver records. All forty changed trajectories complete;
none of the eight whole configurations survives. All original inputs are
accepted, illustrating again that acceptance does not establish correct shape.

This separates valid-source-normal eligibility from a sign-invariant 45-degree
source/target normal-axis filter. Source normals rotate with the current ICP
estimate; translations and arbitrary normal signs cannot change compatibility.
The angle rule is geometrically specified before trials, not tuned on results.
The original forward pairs are filtered before the unchanged 80% distance trim.
Observation computes shadow counts but returns the actual original pair list.

Both filters cross ordinary/dense 900/1800 source/retained capacity and original/
centered finite updates, using grid/pooled/residual and the unchanged ordinary
1800-point model. GPU normals remain ten-neighbor. No pyramid, voxel-first model,
new crop, fallback, accepted held pose, or looser acceptance threshold is added.
Every iterator call retains source-field summaries, pair counts and normal-angle
summaries; full original solver tracing stays separate.

Six exact observation pilots must reproduce original F1/M7 trajectories and
raw checks, plus M13's exact ten-neighbor observation solver traces. Forty
changed trajectories retain all 840 required pairs, original/eligibility/angle
comparisons and 420 density-by-filter contrasts. All five-case guards remain.
M17/M18 audits must finish first; any survivor takes longer/handheld Stage B
priority. Reused recordings do not establish independent physical accuracy.

The executing snapshot contains 158 Python/SDK JSON files, with 245 separate
verification-source files. All 80 focused tests pass, including actual iterator/
tracker composition, malformed inputs, failure handling and scope restoration.
The selected M1–M19 suite passes 846 tests, lint and cold typing of 113 files.
These are implementation checks, not scan qualification. Production, firmware,
drivers, dependencies, scanner settings and mesh processing are unchanged.

The distinct [residual-weighting experiment](reliability-robust-residual.md) is
prepared behind M19; it does not change this frozen filter or add a combination
to this phase's roster.

## Completed outcome, 2026-09-09 03:02 UTC

Every cell completes and every required raw pair is checkable: 840 checks.
The 420 original/eligibility/angle comparison rows and 420 density-by-filter
contrasts are fully computable. No threshold, pair roster, crop or pose is
changed after seeing these results.

| Capacity | Filter | Update | Cases passing | Binding failures |
| --- | --- | --- | ---: | --- |
| 900 | Eligibility | Original | 3/5 | A1 loses 1.112 points; C minimum 14.568% |
| 900 | Eligibility | Centered | 3/5 | A2 minimum 29.000%; C 21.306% |
| 900 | 45-degree | Original | 3/5 | A2 minimum 1.804%; C 16.618% |
| 900 | 45-degree | Centered | 3/5 | A1 minimum 0.233%; A2 2.901% |
| 1800 | Eligibility | Original | 3/5 | Static B loses 2.194 points; A2 loses 1.826 |
| 1800 | Eligibility | Centered | 4/5 | Static B loses 2.023 points |
| 1800 | 45-degree | Original | 4/5 | Static B loses 2.239 points |
| 1800 | 45-degree | Centered | 2/5 | Static B loses 2.659 points; A1 loses 1.147; C minimum 50.899% |

The dense centered eligibility variant's A1/A2/C minima are
98.954% / 99.114% / 98.205%; the dense original angle variant's are
97.781% / 98.791% / 99.186%. Those good moving results do not excuse their
stationary regression. Ordinary centered angle filtering passes C at 98.708%
but breaks both other moving controls. These are different whole settings,
not interchangeable per-recording choices or a composite qualified solution.

All 60,296 requested fits complete, without a fit refusal. Eligibility alone
removes only 0.0718–0.3610% of forward correspondence occurrences per cell;
angle filtering removes 0.5511–7.5694%. These count repeated ICP-iteration
occurrences, not unique physical surfaces. Even small correspondence changes
can propagate through the retained history; a low rejection fraction cannot
certify an unchanged trajectory or physically correct source normals.

The archive is sealed: 654 files / 1,330,781,316 bytes, manifest SHA256
`9b8582c9f126b6918ca410af4a6957d76ec69290352a53769bfb8f9d2ddf5210`.
Exact inventory and every hash were verified outside that root at 03:06:46 UTC.
The aggregate through M19 is 495 changed trajectories, 99 whole configurations,
10,395 required/checkable pairs, 80 exact controls and zero survivors.
