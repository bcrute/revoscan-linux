# Reverse alignment as a forward-fit initializer

M21 continues [the reliability roadmap](reliability-roadmap.md).
Evidence: `/home/ben/scans/diagnostics/20260909-reliability-reverse-seed/`.
All six observation pilots reproduce original inputs, decisions, poses, raw
scores and full solver traces exactly. All twenty changed trajectories completed
2026-09-09 04:13 UTC and accept every original input. **Ordinary-900 / centered /
reverse-forward is the first complete Stage A survivor**, passing all five
recordings and all 105 fixed pair checks. Longer/handheld qualification follows
in [a separate Stage B record](reliability-reverse-seed-stage-b.md).

The existing point-to-plane fitter uses target normals and one nearest target
per source. This tests solving that asymmetric problem in reverse first, then
inverting its result to seed the unchanged forward problem. Only the actual
forward result reaches the original tracker gates. This differs from M2's
symmetric objective and M6's reciprocal-correspondence filter.

Both calls use the same original inputs, ten-neighbor GPU normals, voxel 1.5,
and unchanged distance/trim/tolerance/iteration settings. Reversal inverts the
initial transform and swaps source/target; forward refinement inverts the
reverse result. Each stage retains its own true counts and residuals. Maximum
work is two times fifty iterations per requested fit. A reverse refusal is not
hidden by fallback, and a finite unconverged seed does not itself earn acceptance.

Twenty changed trajectories cross both capacities/finite updates and all five
cases, preceded by six exact F1/M7 controls with full M13 observation-trace
checks. Retain all 420 required pairs, matching forward-only differences and
210 density-by-seed contrasts. M19/M20 must finish first; any whole survivor
takes longer/handheld Stage B priority. Reused development data is not accuracy,
handheld, live-speed or thermal qualification.

All 65 focused tests pass, covering exact observation identity, unequal input
counts, nonzero rotation/translation inversions, failure propagation, true
metadata and real numerical/tracker composition. Executing source is frozen in
162 Python/SDK JSON files, with 247 separate verification-source files. No
production, calibration, crop, firmware, dependency, driver, scanner or mesh
change is made.

The selected M1–M21 suite passes 997 tests, lint and cold typing of 117 files.
Implementation checks do not qualify recorded or live scanner accuracy.

## Complete Stage A result

Weakest required raw-pair agreement (%), not coverage or calibrated accuracy:

| Capacity / update | Static A | Static B | A1 | A2 | C | Cases passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 / original | 97.370 | 99.780 | 95.090 | 98.180 | 98.521 | 4/5 |
| 900 / centered | 97.434 | 99.780 | 98.389 | 98.222 | 97.673 | 5/5 |
| 1800 / original | 97.390 | 96.970 | 98.322 | 80.260 | 27.230 | 2/5 |
| 1800 / centered | 97.258 | 97.016 | 98.045 | 34.805 | 98.906 | 3/5 |

The survivor accepts all 2,524 inputs. Its largest original good-pair losses
are 0.501, 0.770 and 0.893 percentage points on A1/A2/C, below the unchanged
one-point guard; static B does not regress. Original-update ordinary capacity
fails A1's guard by losing 3.448 points. Both dense variants regress on static B;
original also fails A2/C, centered fails A2. No per-case settings are selected.

All 420 required pairs and 210 density-by-seed contrasts are computable, with
matching forward-only comparisons retained. Across M1–M21, the audited totals
are 555 changed trajectories, 111 configurations, 11,655 required/checkable
pairs, 92 exact observation pilots and one Stage A survivor.

The execution audit records 30,483 completed fit requests, 60,966 completed
stages, 177,511 iterations in those completed requests and 85 unconverged
reverse seeds. Four additional fit requests retain original insufficient-pair
or singularity refusals; these are candidate-fit refusals, not refused input
frames or process failures. The survivor has one such A2 request. All 162
executing source files remain byte-identical to their frozen snapshots.

The combination matters on these development recordings: centered updates and
reverse initialization together survive where the corresponding single changes
do not. Reused development data, modest regression margins and extra solve work
make independent longer/handheld, live-speed and thermal testing essential.

The evidence root is sealed at 04:34:06 UTC: 561 files / 1,055,471,026 bytes,
manifest SHA-256 `9dfd71d299e7f7e46b9073a79375983f32ff877b6f28c2c4111f133cd5004004`.
The separate Stage B root retains the outside-root full-inventory/hash check.
Never append the sealed M21 root. The last host-only snapshot is 50/51/73°C
GPU edge/junction/memory and 71°C CPU package; scanner temperature is unmeasured.
