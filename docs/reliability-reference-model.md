# Reliability continuation: reference/model normal provenance

8 September 2026. Continues [the roadmap](reliability-roadmap.md) and
[the six-minute slow-turntable diagnosis](reliability-slow-turntable.md).
This round is **rejected as a tracking correction**. No production algorithm,
scanner setting, firmware, driver, dependency or experimental live mesh changed.
Evidence: `/home/ben/scans/diagnostics/20260908-reliability-reference-model/`.

## Question and frozen experiment

The previous D replay accepted 1,609 frames while failing 15 of 40 raw-selected
revisits. Direct same-view fits recovered all 40 pairs in both directions and
from both prescribed starts. That motivates inspecting how accepted poses are
reused to construct the ordinary model reference; it does not prove that the
entire trajectory can be repaired by fitting those returns.

`PROTOCOL.md` declared two steps before testing. R1 reconstructs the exact
ordinary model at every late endpoint in D's existing 40-anchor schedule. R2
changes only its normal provenance: estimate the existing ten-neighbour PCA
normal in each retained native frame, rotate that direction into model space,
and give each unchanged voxel centroid its nearest contributing point's normal.
Exact-distance ties follow original input order; missing normals remain missing.

All source preparation, model point coordinates/order/counts, centroid positions,
reference budgets/selection/preference, solver, convergence, correspondence,
trimming and acceptance gates remain unchanged. Adjacent, keyframe and recovery
candidates retain their original normal path. Normal and point counts are checked
at runtime, the native cache is bounded to 32 entries, and the experiment refuses
more than 3,000 ordinary-model calls. No fitted turntable trajectory or sampling,
weight, threshold or parameter sweep is introduced.

## R1: normal provenance differs, but this is not proof of a defect

Every one of the 40 reconstructed targets preserves the saved reference-frame
indices, original retained raw clouds, saved poses, exact staggered sampling and
voxel coordinates. Raw inputs and source reports are hash-checked. The real
RX 9070 XT FP64 normal provider is verified active.

| Original D checks | Targets | Comparable normal directions | Median of per-target median angular differences |
| --- | --- | --- | --- |
| At least 90% agreement | 25 | 34,488 | 6.163 degrees |
| Below 90% agreement | 15 | 20,878 | 9.232 degrees |

Angles ignore normal sign, since reversing the direction does not change the
point-to-plane equation. All individual directions, missing values and angular
differences are retained. Construction and comparison take 7.54 seconds; no pose
is fitted in R1.

The neighbour source-mixing statistic is explicitly a **representative-owner
proxy**: each voxel is labelled by its selected contributor, not by every point
that entered its centroid. Its per-target means range approximately 71–86%.
Native and mixed estimates also use different spatial neighbourhood density.
Their association with bad sections does not establish causation or show which
normal is more accurate. The measured difference justifies the closed R2 test;
it is not a calibration or accuracy verdict.

## R2: both good controls reject the change

Both original 20 ms A1/A2 controls are replayed in full with GPU normals, the
original 60..300 depth band, twelve-frame settling exclusion, 900-point
preparation and stride one. Each offers and accepts all 628 inputs, with 624
completed ordinary-model normal substitutions and no terminal tracking loss.

The guard recomputes **every original checked raw pair**, first reproducing its
baseline score exactly, then evaluating the changed trajectory on those same
endpoints. A new best pair or improved minimum cannot replace an old check.

| Control | Weakest R2 agreement | Worst original-pair regression | Baseline to R2 at that pair | Verdict |
| --- | --- | --- | --- | --- |
| A1, 13 original pairs | 96.5703% | 1.2990 percentage points, source 104/624 | 99.2556% to 97.9566% | Rejected |
| A2, 14 original pairs | 97.3714% | 1.6572 percentage points, source 17/529 | 99.7736% to 98.1164% | Rejected |

A2's source 9/521 also loses 1.2008 percentage points. Every original pair is
checkable and still exceeds 90%, but the predeclared one-percentage-point
no-regression guard is independently binding. Complete frame acceptance and
high minimum scores do not overturn these failures.

As declared, rejection stops promotion and the subsequent static A/B, moving
C/D, development, old-long and handheld R2 replays. They are **not run**, not
assumed successful. R1's D analysis does not count as an R2 replay on D.

## Post-rejection diagnosis: the effect propagates through reference history

Both entire 628-frame trajectories are compared with their respective baseline,
using the same prepared native cloud at every source index. Preparation point
counts, clipped/deprojected counts, depth band, raw return-health records and
raw input manifests are unchanged.

In both controls, the first numerical pose difference occurs at source 4, the
first model substitution. At that saved state, adjacent and keyframe candidates
are still exactly identical to baseline. Keyframe history first differs at
source 24, candidate reference sets at source 25. Reference sets then differ at
603 of 628 sources in each control. This is an observed propagation through
pose-dependent reference selection, not evidence that the reference policy was
changed directly.

Largest median surface-placement differences from baseline are 7.545 nominal
mm at A1 source 394 and 4.882 at A2 source 267. Baseline intermediate placements
are **not independently measured physical truth**; these are trajectory
disagreements, not certified error magnitudes. They nevertheless demonstrate
that good return scores can coexist with quite different intervening placements.

All 81 saved candidate alternatives across the 27 original pairs are audited.
At the three guard-failing pairs an adjacent alternative scores better than the
selected model candidate, but alternatives have different rankings elsewhere.
They are counterfactuals at the changed run's saved history, not a new closed-
loop selection test. Earlier failed reference-preference sweeps and D's bad
sections, where all alternatives shared the failure, remain binding. No
post-hoc switching policy is presented as a fix.

## Verification and implementation boundary

The helper and profile live only under `tools/`; no GUI or production module
imports them. Non-default diagnostic results always prohibit qualification.
The executing R2 source is preserved separately in `r2-source/`; the final
helper additionally labels its measured rejection in its module documentation.

The new focused tests cover exact model/voxel construction, normal rotation
without translation, sign, ties, missing and malformed values, provider-scoped
cache bounds, context restoration on success and exceptions, non-model
transparency and closed parameter/call limits. The selected broader suite has
**223 passed, one skipped**, no failures/errors, in 42.702 seconds. The skip is
the explicit real-GPU opt-in check, executed separately: **one passed**, no
skips, in 1.013 seconds. Its separate XML preserves that result. Earlier 13- and
48-case runs overlap this suite and must not be added as distinct coverage.

An additional full default GPU replay reproduces **all 628 poses and complete
tracking decisions exactly**, along with the full original coherence report,
preparation and raw-health records. Timing and added diagnostic metadata are
excluded from equality; default model-normal records are null. This checks that
the opt-in harness change has not silently changed baseline behavior.

Ruff, mypy over 80 files, source-snapshot comparisons and earlier evidence-seal
verification are recorded in `execution-summary.json`. No production source
file changed in this round: all 79 match the initial snapshot, and 396 of 397
pre-existing source/tool/test files are unchanged. All five earlier evidence
roots verify with no changed or unlisted files. The one pre-existing tool change adds the explicit
diagnostic profile and its reporting; unrelated owner changes are preserved.

## Roadmap state

R1 and the two-control R2 test are complete; **R2 is rejected**. Registration
remains the release blocker. Neither this test nor the prior matching-reference
fits establish a reliable handheld or full-turn operating envelope. The five
fixed/five handheld release trials, live integration and further endurance
qualification still await a candidate that survives the geometric gates.

The next engineering question is how to preserve independently checked local
geometry when accepting poses changes the references used to judge future
poses. Native model normals alone are not a safe answer. A future correction
needs a genuinely different, predeclared mechanism and must preserve the
original-pair guards; this result does not authorize retuning R2 to the controls.
No new operator scan is needed just to reproduce the known failure.

This round opens no scanner stream and makes no new temperature measurement or
claim about the physical turntable's current state. The last hardware baseline
restoration and temperature observations remain those recorded in the sealed
C/D round. Evidence sealing establishes file integrity, not scan correctness.
