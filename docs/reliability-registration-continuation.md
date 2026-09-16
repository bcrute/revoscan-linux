# Reliability continuation — 8 September 2026

Continuation of [the roadmap](reliability-roadmap.md) and
[the first execution record](reliability-roadmap-results.md).

The subsequent operator-supplied view is recorded in
[the face-B follow-up](reliability-view-b.md); this page preserves the earlier
round's outcomes and its then-current boundary.

Status at 14:30 UTC: **registration is not reliable enough to release**.
Three further closed corrections were rejected. Production preparation, ICP,
tracking gates and 20 ms default remain unchanged. Capture diagnostics and
offline-test isolation have been improved. Staged baseline hardware diagnostics
are independent measurements, not a bypass of the release gates.

Evidence root: `/home/ben/scans/diagnostics/20260908-reliability-registration/`.
Protocols were written before their experiments. The previous two sealed roots
were independently hash-verified unchanged. Final source and test results are
included in this continuation's `evidence-manifest.json`; that integrity seal
does not turn a rejected experiment into a quality-qualified scan.

## D3 / D4: convergence and spatial consistency

The unchanged point-to-plane solver was traced at every credible pair in the
five existing good/development/adverse loop schedules, in both directions, and
at the final eight offered frames in each handheld source. The tracing unit
test requires exactly the original solver result. No reported pose was used as
an invented initial motion prior when that prior was absent from the archive.

Residual convergence can occur while pose motion remains significant. In the
188-frame handheld source, adjacent fit 183 exactly reproduces the recorded
accepted fit: residual 1.36891, 11 iterations, but last surface step 0.78464 and
one further step 1.22673 nominal mm. Raw reprojection agreement is only 43.40%.
Fixed left/right/top/bottom subfits disagree with its full-frame placement by
5.81 / 3.82 / 2.92 / 13.10 nominal mm. Later frames are even less consistent.
Full-frame depth plots preserve the striped, fragmented returns and clipping;
they are not semantic arm/object masks.

The normalized local information matrices and residual-based uncertainty
estimates are diagnostics, not calibrated probabilities. Large half-turn loop
errors were not supported: none of the original audited loop relative angles
exceeded 0.5 radians. These results do not identify firmware as the cause.

Older handheld projects do **not** retain actual exposure/gain/temperature
readbacks. GPU requested is not proof that the GPU activated. The failed
188-frame project's reconstruction-dependent timing metadata is absent, and
recovery-period raw frames were not retained. These limits cannot be repaired
by relabeling archival data as known 20 ms acquisition.

## S1: stricter pose settling — rejected

The closed probe requires both the existing residual convergence threshold and
surface movement below that same threshold. It changes no iteration budget,
correspondence rule, reference policy or tracking gate. It is not enabled by
the application.

| Recording | S1 result |
| --- | --- |
| Handheld pauses | 180 accepted of 186 offered; terminal loss remains |
| Combined handheld/table | Stops at 94 offered, versus 303 originally |
| Good 20 ms A1 | 624/628 accepted; weakest revisit 8.92%, originally 96.50% |
| Good 20 ms A2 | Stops at 366 offered, versus all 628 originally |

The controls reject S1. Longer adverse replays were deliberately not run after
that rejection. More ICP iterations on noisy correspondences are not a fix.

## Fresh static control and S2 sampling — useful diagnosis, rejected correction

The new 320-frame preflight has a stationary-looking raw view, complete unique
frames, verified 20/20 ms and gain 24, no sequence/transport fault and stable
final idle. Three full-frame inspection images retain the complete scene.
There is no independent pose encoder or dimensional calibration.

The ordinary GPU tracker accepts all 320 frames, yet maximum prepared-surface
displacement from identity is **27.3894 nominal mm**. The 20 fixed check pairs
have raw identity agreement at least **99.2376%**, but applying the accepted
poses lowers the minimum to **69.0011%**. This is a fresh example of silent
geometric error, not a throughput or dropped-frame event.

Adjacent raw valid-pixel sets retain a median 98.17% of their locations. The
production compacted-list stride retains only **1.87%** of the same sampled
pixel locations; small return changes shift the sampling across the image.
S2 selects the identical per-frame sample count using one fixed integer rank
per absolute pixel index. It changes no raw data, band, filters or tracker
gates. No rank/seed sweep was performed. Sample-location retention rises to
97.47% and the static result improves sharply:

| Known recording | Production | Fixed-pixel-rank S2 | Decision |
| --- | --- | --- | --- |
| Static maximum apparent surface motion | 27.3894 | 0.1960 nominal mm | Static improvement only |
| Static weakest check | 69.00% | 97.50% | Not moving validation |
| Good moving A1 weakest check | 96.50% | 96.96% | Reject: another pair loses 2.292 percentage points |
| Good moving A2 weakest check | 98.48% | 13.72% | Reject: severe silent error |

All 628 frames are accepted in each S2 moving control. Acceptance did not
protect geometry. The complete A2 preparation audit verifies identical input
hashes and thinning counts; support-plane filtering applies to **zero frames**
in either version. Median post-filter counts differ, 806 versus 764. Pose
difference first exceeds 1.5 nominal mm at frame 101 and reaches 18.7442.
Thus S2 is not promoted and does not advance to long/handheld replays.

## S3: spatially regular fixed sampling — close, but rejected

A ten-frame fixed A2 audit measures 9.98–13.41% of S2 sample points with a
neighbor within two pixels, versus 3.30–10.55% under the original stride.
S3 therefore tests spatial regularity, using one coarse-to-fine grid ordering
of absolute pixel coordinates instead of a random-like ranking. It has no
grid-spacing parameter or seed search, keeps identical pre-filter counts and
changes no raw pixels, filters or tracking gates. The implementation remains
an offline, single-thread diagnostic.

All 320 static frames align; maximum apparent motion is 0.4242 nominal mm and
the weakest fixed check is 97.3701%. Both moving controls accept all 628 frames.
A1's weakest revisit is 98.5705%, with maximum original-pair regression 0.6851
percentage points. A2's weakest revisit is 97.8655%, but pair 113/625 falls
**1.5024 percentage points**, above the predeclared one-point limit. S3 is
therefore **not promoted**, and its long/handheld replays are deferred under
the rejection protocol. A useful improvement on static data does not override
the control gate. This candidate merits fresh evidence, not a changed pass bar.

## Retained implementation: capture evidence, not geometry repair

The GUI now writes `diagnostics/capture-evidence.jsonl` independently of whether
the scan can be reconstructed. It records complete delivery indices, sequence
numbers, host times, payload hashes, canonical save mapping, tracking decisions,
failures, GPU runtime messages and the stream's result. Short-delivery metadata
is appended if the stream returns normally; a stream exception can leave that
part unknown. Host delivery time is not exposure time. Device-setting and chip
readbacks remain explicitly unavailable in this GUI path.

Up to 256 recovery/relocalization raw copies are stored as `delivery_*.raw`,
never `frame_*.raw`. They cannot resolve loss or enter reconstruction. Published
segments carry exact delivery-index mapping, including sliding tentative
buffers. The journal is capped at 30,000 events / 64 MiB. Raw/journal omissions
and write failures are explicit; diagnostics never bypass device cleanup.

The existing validity reader, loss records, gates, models and experimental
no-live-mesh restriction are unchanged. Failed old captures are not rewritten.

A bounded I/O measurement with concurrent tests averaged 0.897 ms per ordinary
delivery/save/decision journal cycle (p95 0.948 ms), and 1.601 ms in the recovery
copy exercise (p95 1.839 ms). The latter retained exactly 256 raw copies and
reported the 64 later omissions. These are filesystem/journal measurements,
not a complete live GUI throughput qualification.

## Acquisition and thermal diagnostics

Exact connected MINI: `A22704522A6F00A64`, 480 Mbps USB link. Every new hardware
run uses 20/20 ms, gain 24, auto exposure off. CPU/NPU flag 75 C, stop 80 C;
these are chip readings, not certified optics/case limits. Quiet STREAMOFF,
fresh-descriptor readbacks, restoration and final idle are checked. First
transport faults terminate the branch without automatic reset/retry.

| Run | Complete/distinct frames | Observed fps | Final chip readings | Lifecycle |
| --- | --- | --- | --- | --- |
| Preflight | 320 | 4.506 | 59.8 / 59.1 C | Verified |
| 5-minute acquisition | 1,351 | 4.517 | 62.0 / 61.3 C | Verified |
| 15-minute acquisition | 4,061 | 4.517 | 65.1 / 64.4 C | Verified |
| 30-minute acquisition | 8,125 | 4.517 | 66.0 / 65.3 C | Verified |

Ten full raw frames per minute are retained, plus all delivery metadata and
regular density/thermal samples. Every minute is compared at fixed initial
pixels, including the full return footprint and its fringes; no registration
or per-window bias subtraction hides change. These excerpts are not continuous
raw retention, moving geometry qualification, or a complete GUI-load soak.

The 30-minute stage contains 339 thermal observations with maximum interval
5.325 seconds; last-five-minute chips span 65.5–66.0 / 64.8–65.3 C. No stale
telemetry stop was triggered. The final raw excerpt, against the **same first
five-minute-stage reference**, has whole-footprint depth-change absolute
p50/p90/p99 of 0.1625 / 0.3875 / 1.1194 nominal mm. Its stable-pixel persistent
loss is 0.307%; full-footprint overall return retention is 99.046%, but fringe
retention is 92.700%. The largest initial connected region retains 99.422%
overall / 99.959% interior / 94.900% edge. Its worst minute's edge retention is
94.449%. No mask was adjusted to discard those losses. Connected regions are
not semantic object masks; this is neither a thermal-causality proof nor a
claim that unchanged chip temperature ensures unchanged optical accuracy.

The scanner is idle after verified 20/20 ms, gain 24 and final USB stability.

Full-frame images from minutes 0, 15 and 29 of the final stage were inspected:
the main view remains stationary-looking, with visible fragmented returns
below the object. These images do not override the measured fringe losses.

An unmarked GUI unit test lacked its fake stream and attempted real device
opens during concurrent testing. Its S_FMT attempts were rejected with EBUSY;
the active captures continued. The interrupted broad run and failed GUI run
remain in evidence. The test now uses the fake driver, and normal tests block
unmocked Python opens of video/media/USB nodes. This is Python-level test
isolation, not a subprocess/FFI sandbox. Treat the affected hardware runs as
annotated diagnostics, not pristine release-validation trials.

## Remaining gates and next physical evidence

None of the corrections survived good-control regression, so no new tracking policy
can enter live or fresh-handheld qualification. The current view is now heavily
inspected development data. Further operating-envelope evidence needs another
physical view, known settings and explicit pause/motion boundaries, with the
camera supported and the turntable initially stopped. Operator repositioning,
handheld motion, five fresh runs per intended mode and calibrated CAD checks
remain deferred. Firmware flashing remains gated by authenticated images,
calibration backup and a proven recovery path; no flash was performed.

| Roadmap branch | Continuation disposition |
| --- | --- |
| Drift diagnosis | Completed D3/D4 plus fresh static and sampling audits |
| Closed correction / good controls | S1, S2 and S3 rejected; no production promotion |
| Evidence and lifecycle integration | Bounded GUI journal and unit-test device isolation retained |
| Independent acquisition/thermal branch | 5, 15 and 30 minutes completed; edge loss reported |
| Live geometry/GUI-load endurance and release | Not qualified; acquisition-only endurance is not a substitute |
| Next independent physical diagnosis | Awaiting a different object face with camera fixed and table off |

The next action is deliberately a new **diagnostic** view, not five fresh
release attempts using a correction that has already failed. Keep the camera
supported and fixed, leave the turntable off, rotate the object to expose a
different face while keeping it fully visible, then report ready. The bounded
capture and fixed analysis protocol is recorded in the evidence root as
`NEXT-PHYSICAL-EVIDENCE.md`; a successful new view cannot erase the existing
moving-control failures.

Final protected full regression: **6,356 passed, 16 skipped**, no failures or
errors, 2,379.682 seconds (`full-guarded-regression.xml`). The final focused
continuation selection, including S3 added after the full suite was collected,
passed **88 tests**. Separate guarded GUI/isolation and mapping/pause/recovery
batches passed 357 and 90 tests. These batches overlap and must not be summed
as independent tests. Ruff passed for all continuation code/tests; mypy passed
for 86 application/diagnostic files; `git diff --check` passed. Production
preparation, tracking, ICP and GPU normals match their prior snapshot exactly.

Twelve full-suite skips were real-GPU opt-ins. A separate enabled run passed
all **13 tests** (those twelve plus one ordinary backend-isolation check) with
no skips, covering the experimental kernels and production normal provider
against CPU reference results. The raw-to-PLY CLI test also passed separately
using today's verified `fresh-preflight/raw/frame_0000.raw`. It checks point-cloud
structure/accounting, not dimensional accuracy, and does not build a mesh.
The other full-suite skips remain explicit: insufficient lag in one synthetic
sweep, an absent optional decompressed-kernel cross-check input, and an absent
specific metric fixture. Separate follow-ups do not rewrite the original XML.

The earlier broad test attempt was interrupted after 4,722 cases (three
failures, 14 skips); it is not a full-suite pass. Its three issues were fixed,
and its record remains alongside the successful protected rerun. The source
snapshot includes preexisting owner changes without claiming them as this
continuation's work. No commit, push, deployment, dependency installation or
firmware flash was performed. The next planned independent physical diagnostic
is awaiting operator repositioning; the overall reliability roadmap is open.
