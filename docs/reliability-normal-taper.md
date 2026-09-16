# Tapered covariance within twenty-neighbor normal support

M24 protocol and evidence:
`/home/ben/scans/diagnostics/20260909-reliability-normal-taper/`.
Completed and rejected at 2026-09-09 06:12 UTC. **Real GPU parity and all six
observation replays pass exactly**, but no changed configuration survives.
All twenty trajectories complete with every original input accepted and all
420 original pair checks available. All 210 density interactions, 420 matched
unweighted-normal20 and 420 original-normal10 comparisons are retained.
M21's Stage B rejection and M22/M23's no-survivor
results are complete; any new unresolved Stage A survivor still takes priority.
Nothing is promoted to production or qualified for physical scanning.

Minimum fixed-pair agreement percentages; passing requires every original
pair and the unchanged good-pair no-regression guard, not just these minima:

| Source capacity / update | Static A | Static B | A1 | A2 | C | Cases passing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 / original | 97.492 | 99.780 | 97.766 | 98.331 | 8.579 | 2/5 |
| 900 / centered | 97.492 | 99.780 | 98.049 | 34.232 | 9.177 | 3/5 |
| 1800 / original | 96.059 | 96.963 | 99.248 | 97.253 | 98.719 | 3/5 |
| 1800 / centered | 96.332 | 96.963 | 99.160 | 98.649 | 50.182 | 3/5 |

Dense static B loses 2.75980 points at an originally good pair in both updates.
Ordinary/original A1 and A2 lose 1.48915 and 1.03669 points; dense/original A2
loses 2.11478. The near-threshold outcomes are retained as failures, not rounded
into passes. Tapering covariance support does not solve the combined
stationary/moving requirements in this closed roster.

The 06:16:01 closeout confirms all 168 frozen executing files unchanged.
There are 30,148 completed fit requests, no refused requests, 74,385 iterations,
and 20,053 actual GPU normal-field/Jacobi dispatches. Across 24,853,973 repeated
normal-estimate occurrences, 24,673,924 are usable and 180,049 are isotropic
refusals; none is sparse or coincident. These are repeated algorithm occurrences,
not independent physical observations or a live throughput benchmark.

The existing twenty-neighbor SciPy support is retained. A fixed squared taper
gives outer-boundary neighbors zero covariance weight, with a weighted centroid
and sequential covariance products. This changes normal estimation, not point
positions, solver correspondence weights, model size or tracking gates. It may
smooth membership changes; it may also discard useful support. The complete
recorded result above does not survive the whole-configuration guards.

All 69 focused tests pass, including independently formed weighted matrices,
rigid/scale transformations, boundary replacement, sparse/coincident/isotropic
refusals, exact observation identity, fatal no-fallback behavior, truthful
adapter reports and full tracker/solver composition across both capacities and
updates. The selected M1–M24 suite passes 1,180 tests, lint and cold typing of
123 files. These are implementation checks, not scanner accuracy.

Executing source is frozen at 2026-09-09 03:59:55 UTC: 168 Python/SDK JSON files,
with 250 separate verification files. A missing test import caused one initial
expanded-test failure; both failed and corrected JUnit reports remain. Explicit
framing branches corrected an intermediate type-check complaint, and inspection
caught the copied CLI argument name before the final CLI test/source freeze.
No production, scanner, firmware, calibration, driver, dependency or mesh change.

The declared real-GPU field parity, six exact M13 normal-20 observation controls
and twenty five-case trajectories have all been executed. Passing arithmetic
and observation controls did not imply passing trajectory accuracy.

The 05:12 UTC parity roster note clarifies the existing closed schedule:
192 raw-source clouds already include both 900/1,800-point capacities, so two
normal profiles require 384 CPU/GPU comparisons, not 768. The original 232-cloud
normal-support record also contains 40 retained-model clouds; those are outside
the protocol's original raw-source scope. No cloud is dropped for its result.

At 05:39 UTC the real RX 9070 XT parity audit completes all 192 raw clouds,
384 fields / 441,026 normal estimates, with exact full field and point identity
equality for both profiles. There are 384 actual FP64 Jacobi dispatches,
52.200 seconds elapsed, no fits and no scanner access. Other CPU audit work is
concurrent; this is not an isolated throughput benchmark or normal ground truth.

The complete M24 archive seals at 2026-09-09 06:17:35 UTC: 606 files /
690,197,039 bytes, manifest SHA-256
`3d6d227b696692506722ef4b9c64de48cc47d8928aabb60885ae66a7be628ddf`.
Future output belongs outside that root; its scripts' default output locations
must not be reused after sealing. Living repository summaries remain editable.
