# The raw versus temporally weighted pose-quality upper bound, on one real capture

`tests/temporal_weighted_replay.py` fuses one recorded capture three times from
the same bytes and asks a single narrow question:

> What does [pose-compensated temporal completion](depth-temporal.md) and the
> [provenance-weighted fusion input](weighted-tsdf-input.md) do at the most
> permissive pose-quality value under one fixed nearest-eight source rule, on a
> capture whose raw baseline is already recorded?

It is a harness, not product code. Nothing in `scanner3d` imports it, pytest
does not collect it, and no export, GUI, live preview, capture path, tracker or
readiness policy reads anything it prints. It is read-only on the capture and
proves it.

**The measured outcome on `20260827-152842` is a rejection, and it is stated up
front.** The candidate arm added field observations and weight - 220,217
accepted temporal samples reached 10,023 voxels the raw arm never observed -
but it did not add extracted surface geometry. The surface came out *smaller*,
with a marginally worse isolated-lump proxy. The declared decision rule rejects
it. See
[what the physical run found](#what-the-physical-run-found).

## The one command

```bash
.venv/bin/python -m tests.temporal_weighted_replay \
    --capture /home/ben/scans/scan/20260827-152842 --limit 12 --backend numpy \
    > /tmp/temporal-weighted-upper-bound-038-final.json
```

`--capture` is **required and has no default**: this replay measures exactly the
capture it is pointed at. `--limit` takes the first N name-sorted `frame_*.raw`
files (12 by default, the prefix [task 037's replay](surface-oracle.md) has a
recorded raw baseline for). `--backend` selects the field and extraction backend
for all three arms; the temporal completion always runs on its own and only
`"python"` implementation.

The report goes to **stdout only**. This module opens no path for writing at
all, so the caller decides where the artifact lands and a run that is not
redirected leaves nothing behind.

Exit codes: `0` ran to completion or stopped at one of its own declared
boundaries; `1` failed part-way; `2` the capture could not be bound; `3` the
before/after fingerprints disagree, which dominates every other code.

## Three arms, one field, one set of bytes

Every arm receives the same original raw bytes, the same registration poses, the
same factory intrinsics and depth scale, the same `TsdfConfig` object and the
same backend.

| arm | how it is fused | what it is for |
| --- | --- | --- |
| `raw` | `TsdfVolume.integrate` over the raw frames | the baseline, and the call a production export makes |
| `weighted_raw` | `integrate_weighted(weighted_raw_frame(...))` | the **identity control** |
| `candidate` | `integrate_weighted(weighted_temporal_frame(...))` | the fixed-rule pose-quality-upper-bound arm |

The field definition is derived **once**, from the raw clouds placed by the
registration's own poses, at the desktop Reconstruct voxel budget
(`scanner3d.gui.RECONSTRUCT_VOXEL_BUDGET`, `128 ** 3`), with
`scanner3d.reconstruction_export`'s own bounds, truncation, `TSDF_MIN_DEPTH`,
`TSDF_MAX_DEPTH`, uniform weight policy, `TSDF_MAX_WEIGHT` and `TSDF_MIN_WEIGHT`
extraction minimum - imported rather than restated.

The identity control is established **before** the completion arm runs at all.
`weighted_raw_frame` weighs every measured sample exactly one and every
no-return exactly nothing, so its field must be the raw field's own bytes. If it
is not, the weighted seam moved the field by itself, no later difference is
attributable to temporal evidence, and the replay stops at `identity_control`
without running the candidate.

Every coordinate is in the **model frame** the poses were fitted in. The
presentation half turn `UPRIGHT_FROM_OPTICAL` is applied to nothing, on any arm,
for the reason [task 037's replay](surface-oracle.md) documents.

## The fixed source rule

`scanner3d.depth_temporal.complete_temporally` accepts at most
`MAX_SUPPORT_FRAMES` (eight) source frames. So each target uses exactly

> the **eight** other frames nearest to it in capture-index distance, ties
> broken by the **lower index**, offered in **ascending index order**.

`nearest_sources(target, frames)` is a pure function of two integers. It cannot
see a raw frame, a pose, a confidence or an accepted sample, which is what makes
the rule checkable rather than merely asserted. It was read off the API ceiling
and written down before any temporal outcome existed, and it is neither
narrowed, widened nor reordered afterwards. Nothing else is post-selected
either: every selected frame is completed, every completion is adapted, and
every adapted frame is fused, in order. A refusal raises out of the loop and is
reported as the result rather than skipped.

The rule is cardinality-strict. Fewer than nine selected frames stops the replay
at `frame_count` before preparation, and `nearest_sources` itself refuses any
call that cannot supply exactly its requested number of other frames. It never
silently turns “eight” into seven or fewer.

## The one deliberate epistemic limit

`reconstruct_sweep` publishes accepted poses and **no calibrated scalar** that
could honestly populate `depth_temporal.PoseRecord.quality`. So exactly one arm
runs, named `uncalibrated_pose_quality_upper_bound`, at quality `1.0`.

Temporal confidence is monotone in pose quality, so that arm is a
**pose-quality upper bound under the fixed nearest-eight source rule and
otherwise fixed frames, poses, policy and fusion inputs**. It is not a global
upper bound on the method or capture: other legal eight-of-eleven source
subsets are not tested. It is not a calibrated physical confidence, it cannot
authorize product routing, and no pose-quality mapping is invented, fitted or
tuned anywhere in this module. The artifact says so in the arm's own name, in
`temporal.meaning`, and in `pose_quality_calibrated: false` at the top level,
in `temporal` and in `decision`.

## The decision rule

Declared in `DECISION_RULE` before the run and applied in this order:

1. `comparator_invalid` - the identity control did not reproduce the raw field.
   The candidate arm is not run at all.
2. `negative_same_input` - no temporal sample was accepted anywhere, **or** the
   candidate field and its surface are identical to the raw ones. Same input,
   same answer: a negative result about this capture, not a failure of the run.
3. `not_comparable` - one arm produced a surface or a measured agreement the
   other did not. No verdict is invented.
4. `rejected_worse_observable_proxies` - a declared proxy got worse. The four
   proxies, fixed before the run: `surface_supported_fraction` and
   `evidence_explained_fraction` (worse when lower), `component_count` and
   `isolated_fraction` (worse when higher).
5. `uncalibrated_upper_bound_candidate` - the candidate changed the field and no
   proxy got worse. A pose-quality upper bound under this fixed source rule,
   and nothing more; it does not assert that surface geometry was added.

## The output schema

Strict JSON, sorted keys, finite (`allow_nan=False`); a report that cannot be
serialised under those rules is reported as `unpublishable_report` rather than
published with infinities in it.

| key | what it holds |
| --- | --- |
| `identity` | the capture path, how it was bound, every selected frame's SHA-256 and the digest of that ordered list |
| `read_only` | before/after size, mtime and digest of **every** file in the capture, and whether they agree |
| `returns` | per-frame raw return counts and fractions, counted from the bytes |
| `registration`, `poses`, `coverage`, `readiness_reasons` | the shipped registration result and readiness gates |
| `resolution`, `field` | the one shared resolution, `TsdfConfig` and extraction minimum, and where each constant came from |
| `measured_evidence` | the bounded raw-only evidence sample every arm's surface is compared against |
| `raw_arm`, `weighted_raw_arm`, `candidate_arm` | per arm: integration evidence, weighted integration evidence, extracted surface, component/isolated-lump proxies, measured-support agreement, and the field delta against the raw arm |
| `temporal` | the arm's name and meaning, the shipped policy, per-frame source sets, declared work budgets, actual work, the full rejection taxonomy, confidence/support/viewpoint summaries and weighted provenance counts and weights |
| `decision` | the outcome, the rule it came from, the four proxies on both arms, and every claim flag at `false` |
| `limitations` | the same list `LIMITATIONS` publishes |

## What the physical run found

The original round-one run, `--limit 12 --backend numpy`, exited `0` in
**165.359779900 s**. Its report SHA-256 is
`49eff62db8a37381869b6fbef27ca55f35865b8ea39cc81a4315e842923366c4`.

Review then corrected only the exact-eight refusal boundary and report
semantics. The final-code replay used the identical capture, frames, source
rule, policy, poses, quality, field, thresholds and NumPy backend, exited `0` in
**168.09 s**, and produced report SHA-256
`c8804359e74cb4cfb889492bcf32b0aa6a6cd2828ae98d2200fa207e836f4f40`.
A recursive comparison found 29 changed leaves: 21 elapsed-time values and the
eight approved prose fields. There were **zero** other numerical, structural or
decision differences. Capture identity, read-only evidence and the rejection
outcome are exactly equal in both reports.

All **116** files of `/home/ben/scans/scan/20260827-152842` were byte-identical
before and after, with no changed, missing or appeared path. The 12 selected
frames' aggregate digest is
`7013246d664098e13080e6454508dc85da141e9859cecfe34e1cb5a02528b679`, which is the
same digest [task 037](surface-oracle.md) recorded for the same prefix.

All 12 frames registered (0 revisits, unoptimised) and passed every readiness
gate. Return fractions ran 0.395602 to 0.484820. The shared field is voxel
`1.0896211626245729`, truncation `3.268863487873719`, a 126x116x136 grid of
1,987,776 voxels inside the 2,097,152 budget after one coarsening step.

**The identity control held exactly.** `values_identical`, `weights_identical`
and `surface_identical_to_raw` were all true: 0 voxels of value or weight
difference across all 1,987,776, and the same 17,912 vertices / 31,000 faces.
Every later difference is therefore attributable to the temporal samples.

**The temporal arm, per frame.** The eight-source rule produced source sets from
`[1,2,3,4,5,6,7,8]` for frame 0 to `[3,4,5,6,7,8,9,10]` for frame 11. Declared
work budgets ran **17,330,643 to 18,842,493** - matching the pre-run figures
exactly, and every one below the fixed 25,000,000 ceiling. Actual work was
8,684,000 to 9,074,550 per frame, 107,182,020 in total against 212,590,002
declared.

Of 1,648,267 holes examined, **220,217 were accepted** and 1,428,050 refused:
1,092,493 `no_posed_evidence`, 319,802 `insufficient_support_frames`, 10,389
`edge_discontinuity`, 5,366 `confidence_below_threshold`, and zero for the other
five reasons. Of 11,533,724 projected observations, 7,050,000 were discarded as
`target_already_measured`, 2,415,612 as `reprojection_too_far`, 764,125 as
`occluded_by_nearer_evidence`, 338,739 as `outside_target_patch` and 96 as
`behind_target_camera`.

Accepted confidences ran 0.3224 to 0.9559 with per-frame means 0.5842 to 0.6171.
Every accepted sample rested on 2 to 4 support samples - the shipped policy's own
minimum and maximum - and every frame used all eight of the sources it was
offered. Viewpoint separations into the target camera ranged from a baseline of
12.94 at 3.73 degrees to 96.87 at 28.28 degrees for frame 0.

**What the weighting offered and what reached the field.** The candidate frames
carried 1,423,733 measured samples, 220,217 temporally reconstructed ones, 0
spatially excluded and 1,428,050 missing, for an offered weight of
1,490,053.4906600432 - 66,320.49 more than the control's 1,423,733.0. Reaching
voxels, the contribution split as 2,047,995.0 measured and
**81,773.05252288448** temporally reconstructed, with exactly zero from either
excluded provenance. The field's own weight delta against the raw arm was
81,773.05256576836, which agrees with that contribution to a relative
5.2e-10 - two independent accumulations of the same quantity.

**The field really moved.** 10,023 voxels gained an observation the raw arm
never had and none were lost; 160,355 voxels changed weight and 32,788 changed
value, at a maximum absolute value change of 1.0 and weight change of
2.956395149230957.

**The surface got slightly smaller.** 17,897 vertices and 30,975 faces against
the raw arm's 17,912 and 31,000: -15 and -25. Both surfaces have 23 connected
components and the same 10 isolated components holding the same 35 points.
Measured support - internal consistency against the capture's own raw returns,
which is support and not accuracy - was essentially unmoved: supported fraction
0.9979274611398964 raw versus 0.9979285344381149 candidate, and an identical
explained fraction of 0.9834001144819691 on both.

**Decision: `rejected_worse_observable_proxies`.** The single proxy that got
worse is `isolated_fraction`, 0.0019539973202322464 to 0.001955635022629491.
Stated precisely: the isolated-point *count* did not change at all; the fraction
rose only because the vertex count it is divided by fell. That is a real
observation about this run and it is exactly what the declared rule was written
to catch, so the rule is applied as written rather than adjusted around the
number it produced. The honest summary is that on this capture the arm added
field observations and weight while extracting slightly less surface. The
supported fraction rose marginally, the explained fraction was identical, and
neither component count nor isolated-point count improved. The mechanical
rejection is solely the isolated-fraction denominator effect, not a new
isolated point.

## What this cannot say

* **No truth model exists for a physical capture.** No coverage of a truth
  surface, surface accuracy, dimension error, edge coverage, thin-feature
  survival or calibrated roughness is measured or claimed. The mesh extent is
  never called a known dimension, and no plane or edge region is chosen - the
  harness has no region option at all, so none can be picked after seeing the
  scan.
* **Measured support is not accuracy.** Both clouds live in one coordinate
  system, so a uniformly wrong scale is perfectly consistent in it.
* **No number has a physical unit.** Every length is in the capture's own depth
  units, whose physical meaning this repository has not established.
* **The poses were fitted by this run.** No pose accuracy is measured, and a
  pose the registration got wrong and was confident about is fused as ordinary
  motion by every arm.
* **The quality-1.0 arm is a fixed-rule pose-quality upper bound, not a
  confidence or a global method bound.** See
  [the one deliberate epistemic limit](#the-one-deliberate-epistemic-limit).
* **Nothing is routed anywhere.** No export, live preview, GUI, capture, tracker
  or readiness policy reads this module, and no threshold in the repository is
  changed by anything it prints.

## The direct tests

`tests/test_temporal_weighted_replay.py` runs the whole harness offline on a
small synthetic asymmetric capture at a narrowed test-only voxel budget, so the
real full-frame temporal completion, the real weighted adapters and three real
fusions are all exercised without the physical capture. It catches raw-control
inequality, a model/presentation pose mismatch, any source set other than the
fixed nearest-eight/lower-index rule, source or result post-selection, an
unlabelled or falsely calibrated quality 1.0, capture mutation on every exit
path, unsupported truth or accuracy claims, and any import or call that could
route the candidate into a product path or open something for writing.

## Verification

Final direct-work verification on 2026-08-31:

- 51 focused temporal/weighted replay tests passed;
- 110 neighboring surface-replay, shell-replay and public-CI tests passed;
- Ruff format/check and strict mypy over 209 source files passed;
- the frozen full collection contained 5,717 tests and completed with **5,714
  passed, 3 skipped, 0 failed**;
- `git diff --check` passed and the pre/post full-suite porcelain digest stayed
  `d87d70f68396daf1be90ce81de1dc8b6087a933bb3de60754fe97590fcb76333`;
- public CI remained GitHub-hosted and its workflow/test SHA-256 values stayed
  `2f280fe580851d3eda0828b8f47f5a78827c7c7724568272fda5424db6986191`
  and
  `cb80bea5b4370bf55ba60809fdb67b944df19921907b60aacb6d44d589d68af6`.
