# M23: plane and 3-D correspondence residuals together

Declared during M20 observation pilots, before M20–M23 changed trajectories.
Evidence: `/home/ben/scans/diagnostics/20260909-reliability-mixed-metric/`.
This is a diagnostic hypothesis, not an adopted tracking setting.

Completed result, 2026-09-09 05:40 UTC: all six observation controls reproduce the
original inputs, decisions, poses, preparation, full solver traces and fixed
raw checks exactly. All twenty changed trajectories finish with complete input
acceptance and all 420 required raw pairs checkable. **No whole configuration
survives.** All 210 density interactions and 420 matched plane-only comparisons
are retained. M21's full
Stage B rejection and M22's no-survivor audit are bound into the new plan.
No live speed or independent accuracy claim follows.

Minimum fixed-pair agreement percentages (A/B are stationary; A1/A2/C moving):

| Source capacity / update | A | B | A1 | A2 | C | Cases passing all guards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 / original | 99.238 | 99.780 | 21.855 | 9.863 | 11.773 | 2/5 |
| 900 / centered | 99.238 | 99.780 | 6.956 | 10.412 | 13.601 | 2/5 |
| 1800 / original | 97.947 | 97.838 | 84.881 | 97.602 | 34.186 | 1/5 |
| 1800 / centered | 97.947 | 97.896 | 29.696 | 96.796 | 38.436 | 1/5 |

Dense B exceeds the one-point good-pair loss limit by losses of 1.55446 and
1.49670 points. Dense A2 also fails that guard (1.76517 and 2.57144 points),
despite minima above 90%. These are complete five-case decisions, not a selection
of favorable minima. All moving cases fail at least one original guard.

The closeout verifies all 166 frozen executing-source files still match and
retains all 73,920 completed solver-step records, with no failed step record.
Exactly one computed mixed-objective increase occurs, only
`1.5178830414797062e-17`; the other 73,919 are nonincreasing. Plane cost rises in
7,358 steps and point cost in 40,030, as the mixture permits trading them off.
These counts do not establish physical error causes or a descent guarantee.
They do not support blaming the large trajectory failures on widespread
increases of the tested combined objective.

Keep the original nearest-neighbor pairs and trim, but minimize the sum of
squared target-plane residual plus **0.1 times squared 3-D displacement**.
The latter gives tangential mismatch some influence. The fixed coefficient is
a predeclared trade-off, not measured noise or a tuned confidence estimate.
This differs from pure point-to-point and M5/M16's two-normal covariance costs;
their failures remain recorded. No new correspondence or pose prior is added.

Before solving the mixture, retain the original unweighted plane information
singularity test. A featureless plane must still be refused, despite the extra
3-D term. Both finite updates, original residual/overlap/travel/reference gates,
ten-neighbor GPU normals, 1.5 voxel size and existing point budgets remain.
The original iterator still reports plane RMSE on its last selected pairs after
the update; it does not silently report the new mixed objective as old RMSE.

The closed roster is six exact observation controls then twenty changed
trajectories across both capacities/updates and all five cases, retaining 420
raw checks and 210 density-by-mixture contrasts. M21/M22 and survivor priority
come first. No production, scanner, firmware, calibration, crop or mesh change.

All 53 focused tests pass; the selected M1–M23 suite passes 1,111 tests, lint
and cold typing of 121 files. Executing source is frozen at 03:20:20 UTC in
166 Python/SDK JSON files, with 249 separate verification-source files. The
expanded 32-test iterator/adapter selection passed after fixing
a fixture that named nonexistent `_pairs`/`_trim` helpers. The corrected test
observes the actual last solver pair set and independently recomputes plane
RMSE. The original four failed fixture cases remain in their JUnit artifact.
No quality-qualified result exists; recorded execution is tracked above.

At 04:45 UTC, an additive eligibility amendment/new driver is prepared. It
requires a complete, exact, explicitly rejected M21 Stage B disposition and
still blocks on any M22 survivor. Original driver/protocol/settings remain
unchanged; 17 tests pass for the identical disposition helper used by M22.
Neither partial failed scans nor absent scores clear the handoff. M23 has not run.

Final archive seal, 2026-09-09 05:43:50 UTC: 582 files / 858,300,414 bytes,
SHA-256 `f118825b03843bd28e433c8f367fe7b4b3d925ca90642a826294c36fa20a09e3`.
Exact inventory and every hash verify outside that root at 05:46:04 UTC, under
M24. No later helper output should be written into the sealed M23 directory.
