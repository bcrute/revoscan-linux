# Reliability follow-up: slow turntable C and extended D

8 September 2026. Continues [the roadmap](reliability-roadmap.md) after the
operator started the turntable at its slowest available setting. This is
development diagnosis, not release qualification. Evidence root:
`/home/ben/scans/diagnostics/20260908-reliability-slow-turntable/`.

## C: clean capture, encouraging production loop, rejected samplers

The exact MINI `A22704522A6F00A64` was unowned before acquisition. Verified
20/20 ms, gain 24, auto exposure off; unchanged settled-post lifecycle, quiet
STREAMOFF boundary, fresh-descriptor restoration and final idle.

- 14:54:05.98–14:56:46.82 UTC including setup/shutdown observations.
- All **640 deliveries complete, kept, hashed and distinct**; no detected
  sequence drops, short frames or transport fault.
- **4.51153 fps**; median gap 375.18 ms, with approximately 57–387 ms extremes.
  This remains the uneven baseline cadence, not a faster firmware mode.
- 29 chip observations; maximum/final **61.6 / 60.8 C**.
- Nine predeclared full images show distinct rotating faces. Lower/support
  fragmented returns remain in the evidence. No full-frame semantic mask or
  object-only coverage claim was introduced.

All three frozen GPU replays offer and accept all **628** inputs after the
fixed twelve-frame settling exclusion. The AMD RX 9070 XT FP64 normal shader
is verified active; correspondence search is SciPy. Existing 60..300 band,
900-point target, default tracker and stride one are unchanged. No fitted
turntable axis, enforced closure or starting-view offset.

| Preparation | Original audit: weakest of 15 credible pairs | Common audit: weakest of 23 credible pairs |
| --- | --- | --- |
| Production | **95.3936%** | **96.5137%** |
| Frozen S2 pixel rank | **21.8060%** | **21.8060%** |
| Frozen S3 hierarchical grid | **17.0824%** | **17.1324%** |

All comparisons use the unchanged 1.5-nominal-mm tolerance. The original audit
requested 25 anchors and found 15 credible revisits. The new common audit
retains all 40 predeclared raw-image anchors: 23 credible/checkable, 17 not
credible, no missing accepted poses or unprojectable credible pairs. Its
greedy endpoint separation leaves only **22**, below the required 25. Thus
production's common-audit verdict is **inconclusive**, not qualified, despite
all observed scores exceeding 90%. Both samplers are rejected, independently
of their earlier binding good-control failures.

Raw matching finds 503/504-frame revisit lags; the common checks span
111.20–111.59 seconds, median **111.53 seconds**. Full first/middle/last and
worst-pair pictures show the same distinctive object features returning.
This supports approximately one turn per 111.5 seconds for this recording,
not an externally measured angular velocity or pose. The weakest common raw
identity score is **98.9498%**. The two audit schedules remain separate;
the common schedule does not erase production's lower original check.

## C mechanism checks

The existing candidate audit checked all three saved alternatives at every
original revisit: **45 candidates per preparation**, 135 total. All scores
are checkable. Production alternatives range 90.88–99.89%. Neither failed
sampler contains a 90%-agreement alternative at any of its 15 checked pairs:
S2 ranges 21.28–41.94%, S3 16.80–41.51%. Changing reference preference at these
recorded states therefore cannot repair their shared error; this is a saved-
state counterfactual, not a new closed-loop selection experiment.

All 628 common-source trajectory comparisons are retained. Relative to the
production trajectory, S2's median surface discrepancy peaks at **21.88 nominal
mm** at source 355, S3's at **30.20** at source 353. Both are much closer at
source 128 (0.13 / 0.33), with growing disagreement by sources 256 and 320.
Production is a comparison trajectory, not independently known intermediate
truth. Static sampling improvements did not survive rotation through the
other faces; no sampler was promoted and no condition-number cutoff added.

The first one-off trajectory adapter called a numeric property as a function
and failed before processing a frame. Its original source and failed report
are retained. The corrected v2 completed all 628 comparisons; no scanner
operation or tracking algorithm was changed to fix that reporting error.

## D: longer coverage reveals production's hidden failure

The six-minute follow-up ran under `extended/PROTOCOL.md`, declared before
acquisition, with the same exact device, settings and verified lifecycle.

- 15:07:21.73–15:13:43.17 UTC including setup/shutdown observations.
- **1,621 complete, kept, distinct and hashed deliveries**, no detected sequence
  drops, shorts or transport fault; **4.51659 fps**. The 360-second cap ended
  acquisition normally before the 2,400-delivery maximum. The report explicitly
  records `stopped_early: true`, reason `duration budget reached`; this is not
  tracking loss or evidence of 779 dropped frames.
- 70 chip observations; maximum **62.9 / 62.2 C**, final **62.5 / 61.7 C**.
- Nine full images show continued rotation. Image matches include approximately
  496–500-frame single returns and 993–994 / 1,491–1,492-frame later returns,
  consistent with roughly three turns over six minutes. No exact angular or
  speed-stability measurement is claimed.

Production accepts **all 1,609** settled inputs with no terminal tracking loss,
but geometry **fails**. The original 25-pair audit reaches only **15.3518%** at
its weakest pair. The common 40-pair audit finds every anchor credible and
checkable, 38 endpoint-separated pairs, and **15 failing pairs** below 90%.
Its weakest score is **15.0215%** at source 330/827. No pose, credible match or
failed check is omitted. All common raw identity scores exceed **98.9969%**.

Some other views score near 100%. The short C run therefore cannot qualify the
intervening views it did not revisit: a good return to one face can coexist
with severely misplaced sections elsewhere. Full first/middle/last, first
failing and worst-pair pictures confirm the distinctive features really return.
No object/support cropping or post-hoc pair substitution explains the result.

The original 25-pair saved-candidate audit contains 75 alternatives. At all nine
original below-margin pairs, **none** of adjacent/keyframe/model reaches 90%,
including refused alternatives. Simply switching the preferred saved candidate
cannot repair this history.

## D: the same-view solver succeeds with the matching reference

The subsequent `extended/TRACE-PROTOCOL.md` freezes 40 pairs, both directions
and two starts (identity and the recorded relative pose): **160 fits**, all
completed with verified GPU normals and unchanged solver settings. Every raw
projection after fitting exceeds 90%; the weakest is **90.6128%**. All local
information calculations complete, and no support plane is removed from the
80 prepared inputs.

At the worst common pair 330/827, the recorded placement agrees at 15.02%.
Direct fitting to the matching earlier raw cloud reaches **99.54%** from that
same bad initial relative pose in five iterations. Identity-start fitting gives
99.60%; the reverse-direction results also exceed 98.9%.

This points toward accumulated reference/model inconsistency, not an inability
to align that same-view raw pair or insufficient iterations in this particular
fit. It does **not** prove every intermediate raw frame is geometrically sound,
identify which global pose is correct, or supply a validated correction. These
fit pairs are diagnostic inputs, not held-out validation. No fitted pose was
inserted into the recording. Earlier rejected loop-graph corrections, sampling
changes and reference-budget/preference sweeps remain rejected.

## Processing speed and arrival timing

The serial D offline replay measures preparation + tracking mean **93.86 ms**,
p95 **137.32 ms**, maximum **163.64 ms** over all 1,609 inputs. Preparation's
mean is 17.25 ms; tracking's 76.61 ms. This is about 10.65 frames/s of that
processing work, while acquisition supplies about 4.52 fps. Total replay time,
including other work such as coherence, is 162.08 seconds. These measurements
exclude live capture/rendering/UI/journaling and do not qualify sustained GUI
throughput. Fast computation and complete acceptance did not protect geometry.

C's separate, predeclared timing audit retains all 627 adjacent pairs. The
313 short-gap pairs have median arrival spacing 60.43 ms, raw median depth
change 0.220 nominal mm and fitted surface travel 0.653. The 314 long-gap pairs
have spacing 382.62 ms, raw change 0.140 and fitted travel 0.420. No middle-group
pairs or missing raw comparisons occur. Arrival gap and raw change correlate
negatively (-0.697). This contradicts naively scaling physical motion by host
arrival gaps; exposure timestamps remain unknown. No time-scaled prior was
introduced on this evidence.

## Implementation and limits

The new offline `scanner_frozen_revisits.py` preserves every requested raw
anchor, even when the original matcher cannot establish a period. It applies
one raw-selected schedule to each replay, reports missing/uncheckable poses,
and never lets an unseparated bad pair disappear from the verdict. Endpoint
separation is an engineering count, not proof of statistical independence.
The tool never authorizes scan qualification. Its optional frame count uses
the existing bounded input validator; C's exact original executing source is
also retained. Fourteen focused tests pass. The broader selected suite has
**128 passed, one skipped**, no failures/errors (129 cases, 7.835 seconds).
The skip is the existing synthetic-period/default-minimum-lag case, not a
physical test passed by assumption. Ruff, mypy (79 files) and diff whitespace
checks pass. Previous full-suite results remain separate; counts overlap.
All 395 pre-existing source/tool/test files compare exactly with this round's
initial snapshot. All files in the four earlier sealed evidence roots verify
unchanged, with no unlisted additions. `execution-summary.json` retains these
checks and the earlier manifest hashes.

Production preparation, tracking, ICP and GPU normals remain unchanged.
The earlier 27.39-nominal-mm stationary drift on face A is not fixed by C's
moving-loop result. Handheld validation, live GUI service quality and release
repeatability remain unqualified. No firmware, driver/dependency change,
experimental mesh, commit, push or deployment was performed.

## Roadmap boundary

Acquisition, common-pair evidence handling and this closed diagnosis round are
complete. **Release is blocked at registration, not at hardware permission or
GPU throughput.** Neither production nor S2/S3 supports the required reliable
moving envelope. Do not start the five fixed/five handheld release trials or
repeat thermal soaks as a substitute for a surviving registration correction.

The next engineering prerequisite remains a method that preserves local
geometry while handling inconsistent reference history, with independent
validation throughout a loop rather than only at a rejoined face. The D traces
make this more specific: matching raw references can recover the bad pairs,
but fitting them alone is not a validated repair of intervening poses. No new
correction is claimed or silently activated. Fresh operator motion is not
needed merely to reproduce the failure; the complete C/D recordings are now
available for the next evidence-led implementation experiment.

The scanner is idle and baseline restoration is verified. The physical
turntable was left running by the operator; software did not stop its motor.
No more acquisition is scheduled in this diagnostic round, so it can be turned
off. Source snapshots, failed adapters and all reports are preserved; the final
manifest seals file integrity, not scientific correctness or scan qualification.
