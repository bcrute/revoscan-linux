# Larger image-local plane feasibility review

Evidence: `/home/ben/scans/diagnostics/20260909-organized-plane-review/`.
Declared September 9, 2026 before implementation and raw analysis. This follows
the [immediate-gradient review](reliability-organized-normal-review.md), which
was less repeatable than the existing PCA fields at a much smaller support
scale. The present study fits no tracking pose and is not M27.

Use all earlier 192 raw clouds and unchanged representative pixels. Three
fixed square windows (5x5, 11x11, 21x21) fit an unweighted camera-space plane
to original, in-band depths within 3 nominal mm of the center. Require 75%
window support, a nondegenerate fit and <=0.25 nominal-mm perpendicular RMS.
No holes are filled, depths changed or old normals substituted after refusal.
Full details and numerical/availability tests precede real-input execution in
the evidence protocol. These thresholds are engineering hypotheses, not noise
calibration or an optimization against the recorded results.

All eighty earlier stationary frame-pair occurrences and common pixels remain
in scope. Compare new and original direction changes on identical joint
subsets; count every unavailable estimate. Stability is not physical normal
accuracy, and a chosen low-noise radius is not a qualified tracking algorithm.
The initial 36 focused tests and expanded 73 tests pass. The broader selected
suite passes 1,617 checks, with clean lint and cold typing of 132 source files.
This is not the entire repository suite. The source-frozen raw review starts
after those results; the separate audit protocol fixes full support/identity
checks and every-seventeenth-plus-last scalar/SVD comparisons before numerical
results are read. No mechanism is promoted from these implementation checks.

The 175-executing/218-saved-source run completes at 13:06:34 UTC in 21.73
seconds: 661,539 window-estimate occurrences, 56,703,167 compressed output
bytes. All 96 raw frames and original representative deprojections match.
Available estimates, out of 220,513 per radius: 202,541 at radius 2; 151,320
at radius 5; 66,827 at radius 10. This is repeated algorithm work, not 661,539
independent physical observations.

The independent audit completes at 13:09:13: all 661,539 support/identity
checks pass; its fixed roster samples 39,729 occurrences, of which 30,752 have
enough support for separate scalar/SVD fitting. Maximum centroid/eigenvalue/
normal-component/RMS differences are 1.71e-13, 1.16e-14, 1.27e-14 and
1.03e-13 respectively, within the declared tolerances. It does not independently
recompute every fitted normal's arithmetic.

All eighty fixed stationary comparisons complete at 13:11:04, retaining every
one of the 81,587 original common-pixel occurrences. The following median
angles use each radius and PCA10 on that radius's *same* joint subset:

| Case/capacity | 5x5 / PCA10 | 11x11 / PCA10 | 21x21 / PCA10 |
| --- | ---: | ---: | ---: |
| A/900 | 2.648 / 0.821 deg | 0.620 / 0.718 deg | 0.195 / 0.614 deg |
| A/1800 | 2.133 / 0.865 deg | 0.520 / 0.732 deg | 0.193 / 0.739 deg |
| B/900 | 2.122 / 0.733 deg | 0.493 / 0.619 deg | 0.194 / 0.588 deg |
| B/1800 | 1.850 / 1.007 deg | 0.434 / 0.841 deg | 0.187 / 0.882 deg |

The 21x21/PCA20 same-subset comparison is about 0.187–0.195 versus
0.414–0.553 degrees. Larger support is more repeatable where available,
but lower availability and possible bias prevent a direct tracking conclusion.
Full per-radius and all-six-method joint subsets, counts and quantiles remain
in `stationary-plane-comparison.json`; do not compare different columns as
though they had identical subsets.

A separate `HYBRID-PROTOCOL.md` now declares four fixed organized/PCA10
combinations and explicit switch accounting, before their numerical review.
That pending diagnostic preserves original PCA10 availability and tests whether
fallback/blending loses the stationary benefit. It fits no pose and does not
implement transport through a moving fused model.

The fixed hybrid review completes at 13:20:42. Its 39 focused tests and the
expanded 1,656-test selected suite pass, with clean lint and typing of 133 files.
All 326,348 hybrid availability checks exactly preserve original PCA10; this
is four repeated combinations of the original 81,587 pixel occurrences.
Every combination improves aggregate median/p95 variation in all four strata,
but every combination worsens p95 in its hard-switch subset. For example,
full radius-5 replacement changes switch p95 from 27.79 to 53.22 degrees in
A/900 and from 15.24 to 51.78 in B/900. Half blending reduces, but does not
eliminate, those tails. No aggregate median is allowed to hide that result.

`hybrid-review.json` retains every angle, both endpoint modes, all profiles,
baseline/changed quantiles and the stable/switch subsets. A separately declared
`CONFIDENCE-PROTOCOL.md` now tests smooth support/RMS/angular-agreement tapering
with the same four global radius/maximum-weight choices. The coefficients are
fixed engineering hypotheses, not measured noise or an accuracy claim. That
new review is pending; neither hybrid has been promoted into a tracker.

The confidence implementation's first 36 focused checks retain one failure:
usable status with no direction must be refused before invoking the fallback
helper. An explicit consistency check fixes it; all 36 runtime tests then pass.
The first broad verification passes 1,692 runtime tests but fails four typing
checks around the optional RMS value. Its source and verification remain
intact. After that process exits, an explicit numeric-type guard/cast correction
passes focused typing and all 36 checks. A separately named broad rerun gates
the recorded-data confidence review. Thresholds and numerical rules stay fixed.

The corrected broad run passes all 1,692 tests, clean lint and cold typing of
134 files. Confidence review completes at 13:34:48 with all eighty pairs and
326,348 availability checks exact. All four profiles improve median variation
in all four strata; aggregate p95 is lower or equal, not uniformly strictly
lower. Large added switch penalties mostly collapse toward the original PCA10
level. Small regressions remain: e.g. radius-5/full A/1800 switch p95 is
17.9641 versus original 17.9460 degrees. That is retained, not rounded away.

For the full-maximum-weight radius-5 confidence rule, aggregate median/p95
direction changes are:

| Case/capacity | Original PCA10 median / p95 | Confidence median / p95 |
| --- | ---: | ---: |
| A/900 | 0.846 / 18.425 deg | 0.742 / 18.374 deg |
| A/1800 | 0.923 / 13.449 deg | 0.783 / 12.951 deg |
| B/900 | 0.770 / 13.186 deg | 0.715 / 12.378 deg |
| B/1800 | 1.053 / 12.289 deg | 0.932 / 12.032 deg |

This is an illustrative fixed profile, not a selected release policy. All four
profiles, endpoint weights, hard-switch identities and subset quantiles remain
in `confidence-review.json`. No new normal is created where PCA10 was absent.

The independent 13:37:51 audit reconstructs every one of 326,348 profile/pair
occurrences for both hard and confidence combinations, verifies all 652,696
endpoint weights, and exactly reproduces 256 stored quantile arrays. Maximum
hard/smooth angle differences are 9.65e-10 / 2.03e-10 degrees; maximum weight
difference is 7.00e-15. Original availability, hard-category assignments and
hard angles match exactly. This verifies arithmetic and bookkeeping, not
physical normals, pose geometry, FPS or live behavior.

The next independent implementation problem is transporting these raw-backed
directions into a fused multi-frame model. A model voxel may combine several
frames, and a reverse fit swaps source/target roles. Neither may be inferred
merely from equal coordinate tuples. A separate candidate protocol must bind
raw input to accepted frame identity, retain original model points/voxel order,
rotate the selected source direction into the model frame, preserve the pooled
PCA fallback/availability, and distinguish forward model fits from reverse
initialization. This review itself still fits no pose and declares no M27 result.

Closeout completes at 13:43:44: all 439 source/context identities, six completed
reports and ten retained JUnit records (including actual failed checks) audit
successfully. The immutable 13:47:26 seal contains 690 files / 756,138,314 bytes,
manifest SHA-256
`b4f893a7c7e0ac9972c361529b05eebde36b4fd0c07306fa92cc1db0f223e713`.
An outside verifier in `20260909-organized-model-preflight/` confirms the exact
inventory and every hash at 13:55:34. This is local integrity, not an off-machine
backup or physical qualification. Further model-transport work uses that new
root and separate helpers; nothing is appended to this sealed review.
