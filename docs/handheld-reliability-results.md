# Handheld reference-health results — 8 September 2026

The [predeclared plan](handheld-reliability-plan.md) was executed through its
offline evaluation and regression tests. **Do not enable the experimental
reference policy:** it improves stopping behavior on one recording but makes
another fail earlier. The controlled physical sensor comparison is waiting for
the operator to confirm a fixed camera, fully visible object/base, stopped
turntable and idle application. No scanner stream, setting write or firmware
operation was performed in this execution.

## What was tested

The default replay reproduced all 188 decisions and poses of the latest failed
recording exactly. An instrumented CPU replay also matched the prior GPU result
exactly, including candidates, accepted indices and coherence report. Added raw
yield reporting therefore did not change the default tracking result.

The reference audit found 18 accepted warning frames in that recording; 14
became keyframes. Warning geometry appeared in non-adjacent candidates on 123
subsequent offers. This establishes exposure to sparse references, not that they
caused incorrect poses or the eventual loss.

The one predeclared `return-health` diagnostic keeps the same alignment gates,
reference budgets, normal computation and candidate preference. Warning clouds
remain available for adjacent tracking, then are released when superseded; they
cannot become archival keyframes or model/recovery evidence. Every offered
frame and accepted pose remains recorded. The initial anchor is explicitly
exempt. No warning cloud leaked into an archival candidate in either replay.

| Recording | Existing default | Reference-health experiment |
| --- | --- | --- |
| Move–pause, 188 raw frames | 185 accepted; terminal three-refusal streak | 185 accepted; no terminal streak within saved prefix |
| Combined handheld/turntable, 303 raw frames | 284 accepted; reaches end without terminal streak | 204 accepted; stops after 218 offered |

Indices are zero-based in the JSON evidence. The experimental move–pause
refusals were 127, 184 and 187 rather than the default's 185–187. It did not
produce additional accepted views or prove tracking would continue beyond the
saved prefix. In the contrasting recording it stopped on source frames
215–217; the source actually contains another 85 frames, which were preserved
but not offered after the normal terminal-loss limit.

Neither recording provides a credible automatic revisit, so every geometry
verdict remains **inconclusive**, not passed. No higher-quality model or speed
gain is claimed. Concurrent offline checks also make these elapsed times
unsuitable as processing benchmarks.

## Decision and retained changes

Reject the reference-health policy for production. The contrasting failure
triggers the predeclared stop condition; do not tune the thresholds, try a grid,
or run the expensive promotion-only controls to justify this rejected version.
Removing sparse geometry can discard genuinely new viewpoints, and a scalar
raw-yield warning does not establish which surfaces are useful. The existing
15-frame baseline also adapts downward through sustained sparsity: the last
experimental failed frame reports 5.37% raw yield against a 3.03% recent baseline
and consequently receives a `usable` advisory. That word is not an alignment or
depth-quality certification.

Only diagnostic tooling/tests and these documents changed in this execution:

- Closed, explicitly offline `return-health` replay profile, retained to reproduce
  the failed experiment; no GUI or production tracking policy selects it.
- Per-offer raw-return-health and in-band-yield evidence on replay reports.
- Failed-source provenance: the project loss marker or a `tracking-loss.json`
  sidecar now forces quality eligibility false even for a default diagnostic
  replay; source markers and hashes are preserved. Pre-change reports are left
  untouched and explicitly disqualified in the new comparison.
- **67 diagnostic tests passed** and **300 core regression tests passed** in
  disjoint final test selections. Ruff, scoped mypy and `git diff --check` passed.
  The first new-test run had four incorrect test API names; its failed JUnit
  report is retained alongside corrected/final runs, not removed.

No live mesh, live policy change, firmware flashing, dependency installation,
commit or push. Existing unrelated owner changes were preserved.

## Evidence and next step

Evidence root:
`/home/ben/scans/diagnostics/20260908-reference-health/`

`comparison.json` and `audit-*.json` record source hashes, all actual candidate
references (including recovery partitions), raw-source/accepted-index mappings,
warnings, refusals and the rejected promotion decision. Original captures and
sealed reviews were not edited. The evidence manifest includes both original
source captures, source snapshots, plan/results, replays and test reports.

Next is stage 6 of the plan: isolate sensor timing from physical motion using a
fixed-camera, stopped-table reference followed by matched moving-table clips at
20/20 ms versus 14/14 ms, gain 24 and auto exposure off. Verify current settings
and exact device identity before capture; do not infer them from a GUI checkbox
or old recording. Record raw spatial depth loss, matched views, timing readback,
temperature and transport continuity. Do not promote faster timing on FPS alone.
Extended thermal runs and any firmware escalation remain conditional on that
result and the separate safeguards in the plan.

## Subsequent stopped-table preflight

After the operator confirmed the setup, the first 20 ms physical clip was
captured under a separately predeclared stopped-table A-B-B-A protocol. It
delivered 640 unique complete frames at 4.511466 fps, without sequence drops,
short frames or transport errors. CPU/NPU peaked at 60.7/60.0 C. Exact-serial
readbacks confirmed 20/20 ms, auto exposure off and gain 24 before and after;
restoration and final idle checks succeeded.

The framing check paused the block before any 14 ms candidate: base/support
returns reached the bottom boundary in 610/640 frames. The main persistent
component ended at y=387, but the 50%-frequent component reached y=400. This
does not establish that the printed object itself is clipped—support may be
included—but full base coverage could not be confidently verified. The operator
was asked to recenter with clear margins while keeping the table stopped.

The entire preflight is retained and explicitly unqualified for full-coverage
comparison at `/home/ben/scans/diagnostics/20260908-stopped-timing-abba/`.
No complete timing comparison, moving-table test or long thermal qualification
has been performed from this setup. New offline analysis tooling preserves the
first 60 settled seconds, checks every raw hash including extras, reports fixed
pixel changes without alignment, and refuses flagged scene provenance.

## Adjusted-view stopped comparison completed

After the operator adjusted the object, a new 20 → 14 → 14 → 20 ms block
completed without mixing in the flagged preflight. Its frequent-return main
component ends at row 383 rather than the image boundary; the 90%-persistent
component ends at row 369. Occasional sparse support/background border contact
remains documented, not treated as semantic part coverage.

All four clips delivered 640 complete, globally unique frames: **2,560 total**,
with no short/sequence losses or failed transport request. Full-clip rates were
4.511518 → 6.591881 → 6.591905 → 4.511498 fps. All settings/readbacks stayed
at gain 24 and auto exposure off; timing returned to 20/20 ms after every run.
Each known stop-time USB re-enumeration settled at the declared quiet boundary,
using fresh exact-serial descriptors without resets/retries. Maximum CPU/NPU
was **62.9/62.2 C**. No continuous long-run thermal claim is made.

The same first 60 settled seconds from each clip were compared at fixed pixels:

| 14 ms repeat | Overall retention | Core retention | Edge retention | Persistent stable loss |
| --- | ---: | ---: | ---: | ---: |
| B1 | 96.186% | 98.139% | **74.737%** | 1.827% |
| B2 | 96.168% | 98.189% | **73.970%** | 1.889% |

Returning to 20 ms restored edge retention to 98.185% and reduced persistent
stable loss to 0.022%. Both faster repeats fail the unchanged 95% edge criterion
on the declared A1 mask. Loss is concentrated around the base/support fringe:
this is **not** a measurement that 25% of the printed object was lost. No
mask crop or relaxed limit was used to make the faster setting pass.

B1/B2 p90 absolute median-depth shifts were 0.0925/0.1300 nominal mm, while
A2-to-A1 drift was itself 0.1300 mm. Temporal MAD ratios were at most 1.3 times
baseline, within the 1.5-times limit. Drift and region/support ambiguity remain
explicit limitations, not proof of thermal causality or absolute accuracy.

Conclusion: about **46.1% more raw FPS**, but not a loss-free stationary upgrade;
retain 20 ms as the default. Evidence is sealed separately at
`/home/ben/scans/diagnostics/20260908-stopped-timing-centered/`. The next operator
step requested is the lowest steady turntable speed, with the scanner fixed,
for a separate moving comparison. Faster live timing, moving qualification and
15/30-minute thermal qualification remain pending.

## Low-speed moving comparison completed

The operator then confirmed the lowest steady turntable setting, fixed camera
and idle application. The separate moving 20 → 14 → 14 → 20 ms block is retained
at `/home/ben/scans/diagnostics/20260908-moving-low-abba/`. `PROTOCOL.md` fixed the
acquisition and matching choices before A1. All four clips contain 640 complete,
unique raw deliveries with no short frames, sequence loss or failed transport
request. Every lifecycle verifies exact serial, gain 24, auto exposure off,
quiet shutdown and restoration/readback of the original 20 ms timing. Known
stop-time USB re-enumeration remains in the records, not suppressed or called
an uninterrupted device identity across STREAMOFF.

| Stage | Full-clip fps | First-60-settled-second fps | Frames in that window |
| --- | ---: | ---: | ---: |
| A1, 20 ms | 4.511313 | 4.516362 | 271 |
| B1, 14 ms | 6.591887 | 6.587767 | 396 |
| B2, 14 ms | 6.591802 | 6.587696 | 396 |
| A2, 20 ms | 4.511540 | 4.516575 | 271 |

The unchanged GPU-normal/default-tracker replays accepted all 628 settled
frames in each clip, without a terminal tracking stop. A1 has 13 credible
image-matched revisit pairs and A2 has 14, from the unchanged 25-position sample
procedure. Their weakest agreements are **96.503% and 98.476%**, respectively,
against the existing 75%-within-1.5-nominal-mm criterion. A1's image period is
520 frames and A2's 513: approximately 114–115 seconds per revolution, inferred
from raw views rather than an external angular measurement. No fitted axis or
turntable prior is supplied to the tracker.

The 14 ms short clips are under one turn. Their revisit verdict is **inconclusive**,
not passed. Because this duration issue was evident after A1 and before faster
replay/phase analysis, a prospective protocol extension specified exactly one
separate 14 ms, 1,200-frame/180-second-bounded capture. It delivered 1,183 unique
complete frames over 179.138 seconds between first and last delivery, at
6.598281 fps, stopping normally on the time budget. It has no capture loss or
transport fault; original timing restoration succeeded. CPU/NPU peaked at
**64.2/63.5 C**, the highest in this block. This is a geometric-duration check,
not a 15/30-minute thermal qualification. Its full settled replay is recorded
separately as `extended-replay`, without replacing either short repeat.

That extended replay accepted **1,171/1,171 settled frames** without stopping.
The unchanged 25-position procedure found **22 credible revisit pairs**;
three sampled positions did not meet its raw-image credibility test. The
weakest measured pair, accepted indices 244 and 991 (raw deliveries 256 and
1003), scored **76.445%**. This narrowly passes the existing 75% criterion,
only 1.445 percentage points above it; it is not robust evidence for promotion.
The image-derived median revisit interval was 113.307 seconds. The shorter
20 ms controls inspect different views and durations, so their larger margins
do not by themselves prove that timing caused the geometric difference.

A separately labelled post-hoc diagnostic checks exactly the weakest already
measured pair from each checkable replay. It reproduces the saved-pose score,
then measures the raw same-view pair with identity transforms, without fitting
or changing any saved pose. The extended pair scores **99.941% at identity
versus 76.445% through saved poses**. A1 and A2 identity controls score
99.933% and 99.697%. This strongly implicates accumulated registration error
in the extended pair's weak agreement, rather than a raw-image disagreement
of the same magnitude. Identity is not independent exact-pose ground truth,
and this diagnostic neither repairs the scan nor replaces the existing gate.
The source hashes, selected indices and both projections are preserved in
`identity-control.json`.

### Phase-matched raw depth, including omitted views

There is no stationary averaging of the moving data. Sixty A1 anchor views are
chosen by one-second-bin centres in its first settled minute, not by image
quality. Each is matched to all settled frames in each other clip using the
predeclared missing-aware raw-depth descriptor, without pixel alignment/warping.
The full 60 rows, scores and reasons are retained for every comparison.

**A2 matches 60/60 anchors, B1 46/60, B2 35/60.** Fourteen and 25 candidate views
are therefore omitted from the respective matched-subset summaries, not from
the record. Many nearest frames sit at clip boundaries, consistent with short
angular coverage; others exceed the declared missing-aware score limit even
away from those boundaries. This does not separate absent orientation,
occlusion and corrupted depth by independent ground truth. Do not relax the
matchability limit to convert them into measured good views.

The following are median raw same-pixel retention values on each candidate's
common A1/A2/candidate bins, using the identical A1-defined region in every pair:

| Repeat / common bins | Component: 14 ms / 20 ms control | Core: 14 ms / control | Edge: 14 ms / control |
| --- | ---: | ---: | ---: |
| B1 / 46 | 94.62% / 98.40% | 98.35% / 99.93% | 66.60% / 86.46% |
| B2 / 35 | 94.28% / 98.27% | 97.84% / 99.93% | 65.10% / 85.23% |

Median **paired** candidate-minus-control differences are -3.70/-4.03 percentage
points for the component, -1.57/-2.05 for the core, and -18.93/-18.45 for the
edge. These differ from subtracting independent medians. Raw single-frame
edges are noisy and phase-sensitive even in the 20 ms control; do not apply
the stationary temporal-frequency gate directly to this different estimator.

All 2,560 raw inputs are hash-checked, including frames outside the one-minute
windows. Largest-component border counts are 170/0/0/163; all frames have a
component. Inspected raw/matched views show main printed faces inside the frame
and substantial lower base/support-fringe loss at 14 ms. Reduced border contact
is not improved depth. No semantic object mask or full-object completeness
claim follows from these component statistics. The unmatched views and
angle-dependent holes cannot be erased by favorable matched-subset medians.

### Decision and verification

The slow fixed-camera setup supplies two repeatable 20 ms geometric controls.
14 ms provides roughly 46% more raw frames but is not a loss-free depth upgrade.
Retain the existing 20 ms default, tracking gates and GPU-normal processing.
This is not a demonstrated handheld-speed envelope or proof that arbitrary
camera movement is reliable. No live mesh or firmware writes were introduced.
The stopped edge-retention failure still blocks promotion to a long thermal
qualification; the separate three-minute revisit check does not override it.

The only new repository implementation in this moving block is the offline
`scanner_moving_abba.py` comparison and its tests. **78 diagnostic tests pass**
across moving matching, stopped provenance, lifecycle and thermal comparison;
Ruff, scoped mypy with `MYPYPATH=src`, and `git diff --check` pass. An initial
mypy invocation without the local source path reported an untyped installed
module; correcting the invocation resolved it without a code/type-ignore change.
Tracking, coherence, lifecycle and default replay sources match the snapshots
taken before acquisition. No commit, push, installation or GUI policy change.

Across the four short clips and the separate extension, the final assessment
rechecks **3,743 complete raw files, all globally unique**, their capture hashes,
chronological order, settings/restoration and complete settled-replay source
mapping. The scanner is left idle at its original settings. The next software
investigation can use the saved faster full-turn replay as a repeatable
registration-drift case, with both new 20 ms clips and earlier adverse/handheld
recordings as regression controls. No new hand-held capture is needed merely
to reproduce this newly isolated drift. It must not be advertised as a solved
handheld problem, a faster live default, or an independently accurate scan.
