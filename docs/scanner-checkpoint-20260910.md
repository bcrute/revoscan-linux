# Scanner checkpoint — 2026-09-10

Prepared at the owner's request to pause and let Claude take a fresh look.
This is a self-contained record of the actual stop point, not a new experiment
or a requirement to adopt the previous agent's hypotheses. Work directly in this
repository under [AGENTS.md](../AGENTS.md); `.coordination/` is historical only.
Older documents contain checkpoint-specific “next” actions that have since been
completed or superseded. Use this document to locate the current stop point.

## Resume in one minute

- **Objective:** useful, dimensionally trustworthy component scans for placement
  on a **1/10-scale crawler chassis**, using Linux. The Mapper project is out of
  scope. RevoScan's lack of Linux support is an owner-reported constraint.
- **Not yet reliable:** no complete dimension-qualified workflow has passed the
  [agreed acceptance criteria](component-acceptance.md). Cleaner planes, higher
  FPS, accepted tracking frames and passing software tests are not that result.
- **Last completed experiment:** a supported scanner, stopped turntable and a
  more downward view of the existing printed target's two highest staircase
  treads. Capture, both analyses, audits and evidence sealing are complete.
- **Latest finding:** shorter exposure improves local surface shape even on
  identical pixels. Primary spacing is **9.931477 mm** against a **provisional
  9.90 mm** physical difference. This is **not an accuracy/calibration pass**.
- **Pending owner information:** three location-identified underside-to-top
  caliper readings on each of those two treads, plus caliper model/accuracy and
  resolution if known. These repeat readings have **not** been supplied.
- The last request was to **keep the current scanner angle**; no further print,
  immediate reposition or additional capture was requested. The latest user
  request is documentation for a second opinion, not to resume testing now.
- No experiment, hardware control, firmware change or production-code edit was
  performed while preparing this checkpoint. The old **M27 process is still
  suspended**, not finished. Details and preservation notes are below.

Read next: [latest staircase result](component-staircase-downward-results.md),
[acceptance criteria](component-acceptance.md), then the relevant evidence below.
The [learning index](reliability-learning-log.md) and
[practical component history](practical-component-scanning.md) retain the broader
chronology; do not treat every historical proposal as an outstanding task.

## Physical reference: preserve the datum and feature identity

The owner's two existing readings are recorded separately in
[physical-print-20260831-steps-20260910.json](../reference/targets/reconstruction_v1/prints/physical-print-20260831-steps-20260910.json).
The original [physical print record](../reference/targets/reconstruction_v1/prints/physical-print-20260831.json)
and [CAD target manifest](../reference/targets/reconstruction_v1/reconstruction-target-v1.json)
remain unchanged by these tests.

| Actual physical measurement | Reading | Feature / interpretation |
| --- | ---: | --- |
| Underside to second-highest staircase tread | 16.08 mm | CAD `step_10mm_top`; nominal total Z 16 mm |
| Underside to highest staircase tread | 25.98 mm | CAD `step_20mm_top`; nominal total Z 26 mm |
| Difference of those two readings | 9.90 mm | Provisional common-datum arithmetic, not independently certified plane spacing |
| Platform width measured halfway up, X | 91.77 mm | Not an overall/top outline measurement |
| Platform width measured halfway up, Y | 81.74 mm | Not an overall/top outline measurement |

The 16.08/25.98 mm readings are **from the physical underside**, not heights
above the base's upper face. Do not subtract a nominal 6 mm base thickness.
Neither measures the separate pocket plateau. Its nominal total Z is 22 mm;
the actual base-top-to-pocket-plateau height remains a different, unmeasured
quantity. Earlier overall-print readings of 91.90/81.85/55.85 mm and the
12.10 mm pocket depth remain distinct references, not replacements for these.

Only one reading per tread is currently available. Caliper contact positions,
underside flatness, tread parallelism, repeatability and instrument uncertainty
are unresolved. The outstanding request was **near front, centre and near back
of each flat tread top, avoiding rounded edges**, with separate readings rather
than an average. Repetition does not remove systematic instrument error.

A new reference block was considered, but the owner would have to print it.
Its necessity has not been established; the existing staircase was used first.
Only both tread tops need to be visible for this plane-spacing diagnostic, not
the whole object, underside or pocket floor. The second view already completed
the requested reposition. Physical setup has not been visually rechecked during
this documentation-only pause.

## Last completed experiment and what it actually establishes

Canonical report: [second, more downward staircase view](component-staircase-downward-results.md).

Initial device inspection was **2026-09-10 17:22:57 UTC**. Capture closeout was
**17:23:43.842279 UTC**. Evidence sealing completed **17:35:03 UTC**.

- One continuous **20 / 10 / 20 ms exposure** bracket; **frame period remained
  20 ms**, manual mode 0 and gain 24 throughout. This was **not** a 10 FPS test.
- 160 full deliveries, 120 settled raw frames: 40 per stage, zero sequence
  drops/short frames. Observed cadence **4.52635 FPS**.
- Both tread interiors had usable support. Two fixed regions per tread gave
  four pair combinations in each of 120 frames: **480 successful pair fits**.
  All selected pixels returned at 10 ms. ROIs were chosen visually after capture
  and fixed before fitting heights, not selected for agreement with 9.90 mm.
- Factory intrinsics and raw scale were unchanged. Each plane was fitted freely:
  no expected height, CAD registration, forced parallelism, tracking, temporal
  smoothing or fitted scale entered the measurement.

The primary forward distance uses the upper plane's normal at the lower
centroid; the reverse uses the lower plane's normal at the upper centroid.
For nonparallel planes these are different geometric evaluations, not estimates
to average into a preferred answer.

| Exposure | Primary median (mm) | Reverse median (mm) | Median angle between planes |
| --- | ---: | ---: | ---: |
| Initial 20 ms | 9.931875 | 10.374336 | 3.015 degrees |
| 10 ms | 9.931477 | 10.048318 | 0.789 degrees |
| Final 20 ms | 9.933749 | 10.422897 | 3.344 degrees |

At 10 ms the primary values span 9.921429–9.940762 mm. Across the four fixed
pairs, forward medians span 9.919151–9.964656 mm and reverse medians span
9.965691–10.048318 mm. Do not replace the primary pair with whichever one is
closest to the calipers or most parallel.

Primary discrepancy from provisional 9.90 mm is **+0.031477 mm / +0.3179%**.
The agreed 0.3% threshold would be 0.0297 mm on this dimension. Unknown physical
reference uncertainty and patch/direction dependence make this **inconclusive**,
not a near-boundary pass. It does not qualify X/Y scale or the full component.

A separately planned, post-primary common-pixel control kept the exact same
975 upper and 701 lower pixels in every frame. Lower all-point plane RMS was
**0.242680 / 0.027174 / 0.242616 mm** at 20/10/20 ms; retained-point RMS also
improved. Thus a changing missing-pixel mask alone does not explain the local
exposure benefit on this subset. This does **not** identify saturation or a
firmware decoder bug, prove absolute correctness, or justify a universal 10 ms
production preset. The control does not replace the primary accuracy result.

The [first staircase view](component-staircase-results.md) is already complete:
primary 10 ms spacing was **10.243219 mm**, reverse about **10.579105 mm**,
with about **2.50 degrees** between primary planes. Shorter exposure did not
resolve that view's ambiguity. The second view changed angle, range and physical
patches together, so their difference is not an isolated camera-angle experiment.

Verification in the last experiment: 20 existing preflight/lifecycle/geometry
tests and 7 existing reporting tests reran successfully; both raw/result audits
passed. These are **27 existing tests**, not 27 newly devised hardware tests or
independent metrology validation. An early renderer call failed because capture
had not yet written `report.json`; it had no hardware access and succeeded after
capture completion. There was no repeat hardware acquisition hidden by that retry.

## Device and host state: historical versus freshly checked

**Last hardware-verified state, approximately 17:23 UTC:**

- Revopoint MINI serial **A22704522A6F00A64**, USB VID/PID **2207:110c**, port
  **1-1**, observed link **480 Mb/s**. Do not reuse an old USB device number;
  it can change at the quiet-close boundary.
- Original **20 ms frame / 20 ms exposure, manual mode 0, gain 24** restored and
  verified after guarded shutdown. Capture report records completed status,
  verified quiet boundary, stable post-close identity and no transport fault.
- CPU/NPU started at 57.6/56.9 C; maximum including final readback was
  **58.9/58.2 C**. These are not projector temperatures or a thermal plateau.
  No current temperature reading was taken for this checkpoint. The suggestion
  that 90 C is safe was never established; earlier 80 C abort thresholds were
  local test policy, not a manufacturer's rating.
- No manual replug, persistent firmware/calibration write, production default
  change, new tracking method or experimental mesh was introduced by this test.
  The experimental live mesh remains intentionally disabled.

**Fresh read-only host/evidence checks, 21:27–21:28 UTC (16:27–16:28 CDT):**

- Branch **`handheld-and-depth-band`**, HEAD
  **`1e8d7b22b8c2367d7f8178dada3071ae18be81a9`**.
- `git status --porcelain` had **287 entries before this checkpoint's edits**.
  This is a large pre-existing dirty worktree, including production code, tests
  and untracked tools/documents. HEAD alone does not reproduce it. Preserve it;
  nothing was committed, reset, stashed, cleaned or pushed for this checkpoint.
- Targeted process lookup found only M27 among the scanner/capture/analysis
  names checked. No `3D Scanner` GUI window matched the window-list query.
  These are scoped host observations, not a device control read or an exhaustive
  audit of arbitrary processes. Present physical pose and live controls are not
  freshly asserted from them.
- **PID 2009261** remained **`Tsl`**, command:
  `.venv/bin/python /home/ben/scans/diagnostics/20260909-reliability-organized-model/m27/run-campaign.py --phase matrix`.
  It is suspended, not complete; do not automatically send `SIGCONT` on resume.
- Both complete staircase evidence roots passed a fresh **exact file-set, byte
  count and SHA-256 check**. All **449 frozen live and saved source records** in
  the Stage B source freeze still matched at **21:28:02.836677 UTC**.

M27 and the broader interrupted research are described in the historical
[September 9 discussion checkpoint](reliability-discussion-checkpoint.md).
That checkpoint recorded 37 of 80 matrix cells scored, not a completed matrix.
Its unfinished Stage B scaffolding and interrupted inventories are not ready-made
validated campaigns. Keep partial evidence and timing caveats; do not label an
aborted or suspended run complete. This documentation pause did not resume it.

## Earlier results worth carrying forward

These are leads and bounded exclusions, not a claim that the hardware is broken
or that its calibration is known correct.

- **Throughput versus geometry:** [FPS investigation](scanner-fps-investigation.md)
  found existing volatile timing controls could exceed the baseline ~4.5 FPS on
  the same 480 Mb/s link: roughly 9.6 FPS at 9/9 ms frame/exposure, 11 at 8/8,
  and 13 at 7/7, with meaningful depth losses. USB 2 alone is therefore not an
  established explanation for a 4.5 FPS ceiling. A later 14 ms/gain-24 stationary
  candidate reached ~6.6 FPS and a clean five-minute delivery test, but warm
  matched-pixel changes and unqualified moving geometry prevent calling it a
  reliable universal operating point. **10 ms exposure with 20 ms frame timing
  in the latest tests must not be confused with 10/10 ms faster acquisition.**
- **GPU work exists:** [live GPU report](live-gpu-throughput.md) documents GPU
  FP64 normal estimation in the GUI capture worker with parity checks/fallback;
  correspondence search and other work still use the CPU. It is not a fully GPU
  reconstruction pipeline. ~9–10+ FPS preview/acceptance was demonstrated under
  some conditions, but model updates were slower and shape remained unqualified.
  An accepted tracking frame is not independent evidence of correct motion.
- **Host sensitivity is real:** the [discussion checkpoint](reliability-discussion-checkpoint.md)
  records dramatically different apparent motion from the same stationary raw
  data when sampling/model treatment changes. Other views/moving sequences still
  fail. Broad registration sweeps have not yielded a quality-qualified solution;
  do not repeat a large matrix merely because another parameter exists.
- **Exposure plus reconstruction did not solve size:** the
  [full-turn test](component-turn-exposure.md) used 10 ms points with a fixture
  estimate shared from the two 20 ms brackets. It still measured
  **X 91.020044 / Y 81.106533 mm**, short by **0.749956 / 0.633467 mm**
  (**0.8172% / 0.7750%**) against the mid-height calipers. More converged fixture
  fitting and smoother local surfaces did not constitute a scale fix.
- **The can was already physically measured and the old procedure recovered:**
  [replay](can-replay-results.md) reproduced all 16 retained per-frame diameters
  to their historical precision with identical current/historical-source geometry.
  The retained sprayed-far mean was **57.716585 mm** against 58.0 mm; a fixed
  [tail/support diagnostic](can-support-results.md) moved it to **57.713105 mm**,
  not a correction. Coating/contact/reference limitations remain. Exact-pixel
  ideal controls reproduced known diameter closely but did not validate real
  scanner calibration or remove real-data bias. Do not ask for the original can
  measurements as though no notes exist.
- **Visibility and datum mistakes matter:** the
  [pocket work](component-pocket-exposure.md) corrected an earlier “base” ROI
  that was actually turntable. The actual printed base did return depth. Pocket
  floor visibility is pose/exposure dependent. The
  [stationary-reference audit](stationary-reference-readiness.md) found two
  candidate lateral boundaries lacked outside surface support; missing-depth
  borders were not promoted to precise physical edges. This is a limitation of
  that candidate method/view, not proof that all outline methods are impossible.
- **Firmware investigation has not established a needed flash:** the
  [device-side review](firmware-device-review-20260909.md),
  [package review](firmware-package-review-20260909.md) and
  [public review](firmware-public-review-20260909.md) preserve the earlier work.
  The saved `preisp.rkl` container was read twice identically and its structure
  and flagged CRCs checked, but there is no verified CEVA decoder or reversible
  new mode/restoration path from that finding. Stored firmware was not proven
  identical to the executing image; saved files are **not a full recovery image**.
  A YAML serial mismatch was already known; device binary matrices matched the
  relevant stored matrices. It is not new proof of wrong-unit calibration.
  No global scale, arbitrary intrinsic recentering, double distortion correction
  or firmware patch is justified by the staircase result.

## Evidence and safe reproduction

The most recent evidence is **outside Git**. A repository clone alone will not
carry the raw frames, frozen sources or reports below. Preserve these directories
along with the dirty working tree; no separate off-machine backup was made by
this checkpoint.

Latest root:
`/home/ben/scans/diagnostics/20260910-staircase-downward-iRDqsk/`

- 407 files / 76,784,405 bytes, excluding `evidence-manifest.json`.
- Manifest SHA-256:
  `796a3df3b48e79c8a8416606bf6d6312b4cc8f046a24dd860a496c7bf30f67e4`.
- Read `PROTOCOL.md`, `ANALYSIS-PLAN.md`, `COMMON-SUPPORT-PLAN.md`,
  `regions.json`, `step-results.json`, `common-support-results.json`, both
  `*-audit.json` reports, and `closeout.json`.
- Visuals: `stopped-first-last.png`, `tread-regions.png` and
  `tread-regions-detail.png`. Raw frames and capture report are in `stopped-check/`.
- `run_view.py`, `analyze_view.py` and `audit_view.py` adapt the first-view
  code; `common_support.py` performs the fixed-pixel control. Bindings and saved
  sources record exactly what ran. `final-docs/` snapshots precede seal metadata
  and intentionally do not contain this later checkpoint.

First-view root:
`/home/ben/scans/diagnostics/20260910-staircase-check-0aO0EY/`

- 383 files / 72,876,568 bytes, excluding its manifest.
- Manifest SHA-256:
  `baefc490af7adfa4a141e476e3f2d771a4c1b84b98030cb0207605a3a71ce649`.
- Contains `run_check.py`, `analyze_steps.py`, `audit_steps.py`, first-view raw
  frames, tests and the bound geometry/reporting sources reused by the second view.

Related saved evidence roots (details and manifests are linked in their reports):

- `/home/ben/scans/diagnostics/20260910-can-replay-2JXPFA/`
- `/home/ben/scans/diagnostics/20260910-can-support-NKJ6NM/`
- `/home/ben/scans/diagnostics/20260910-stationary-datum-DqRI12/`
- `/home/ben/scans/diagnostics/20260909-component-exposure-turn-3mDF8j/`
- `/home/ben/scans/diagnostics/20260909-component-exposure-fullview-50rD0l/`
- `/home/ben/scans/diagnostics/20260909-pocket-exposure-Fe9kV9/`
- `/home/ben/scans/diagnostics/20260909-reliability-organized-model-stage-b/`
  (`source-freeze.json` and `verification-source/`; 449 bound source records).
- `/home/ben/scans/diagnostics/20260909-reliability-organized-model/m27/`
  (**unfinished**, not a completed/sealed campaign).

Reproduction cautions:

1. **Do not rerun output-producing scripts inside sealed roots.** Many derive
   `ROOT` from their own location and use exclusive output creation. Use a new
   temporary diagnostic root and explicitly bind copied/adapted inputs. Sealing
   is an evidence convention, not permission to assume the filesystem is immutable.
2. Use `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B` when importing saved code
   so it cannot add `.pyc` files to sealed roots. `close_view.verify_sealed(root)`
   is read-only; `close_view.main()` writes snapshots/manifests and must not be
   rerun. The latest full recheck was read-only and did not reseal anything.
3. The guarded acquisition source is
   `/home/ben/scans/diagnostics/20260909-component-exposure-fullview-50rD0l/run_stopped.py`,
   with `quiet_capture.py` and `pocket_geometry.py` in the pocket evidence root.
   Read their bindings and lifecycle tests before another hardware run. The
   established close sequence uses a **six-second quiet boundary** and checks
   stable identity before reopening controls. A first transport fault prevents
   further UVC controls; do not improvise repeated commands on a stale descriptor.
4. Depth is 640 x 400 little-endian Y16, 512,000 bytes/frame; codes 0 and 65535
   are missing/reserved. The existing raw-code conversion is approximately
   **0.004999999888241291 mm/code** despite a misleading legacy constant name
   `DEPTH_SCALE_METRES`. Follow the saved geometry and unit tests; do not silently
   add/remove a factor of 1,000. Factory rectified intrinsics used here have
   fx = fy = 898.6829833984375, cx = 42.43747329711914,
   cy = 192.8326873779297. Unusual cx alone is not a demonstrated defect.
5. The existing environment supplies NumPy/PySide6 and the tested numerical
   stack; plots in these experiments used Qt offscreen, not matplotlib/Pillow.
   No dependency installation or environment update was done for this checkpoint.

## Decision left open

The last selected next diagnostic was the physical tread reference check, not
another exposure/ROI search or a flash. A fresh agent can reassess that choice.
The useful unresolved distinction is between **actual printed-surface/contact
geometry**, **view/exposure-dependent raw depth errors**, and **multi-view pose /
reconstruction errors**. Existing evidence shows sensitivities in more than one
part of the system; it does not yet assign the dimensional error to one cause.

Keep the finish line explicit: below 0.3% on preselected physically measured
observable dimensions, three independent passing acquisitions at a fixed recipe,
a different-component check, required mounting/exterior geometry actually observed,
and a usable Linux capture/save/reopen/export/CAD-import workflow. Reference
uncertainty can make a comparison inconclusive. No criterion is relaxed or newly
passed by this checkpoint. Firmware changes remain a possible later tool, not a
required milestone or a substitute for an isolated, testable explanation.

Checkpoint-only edits: this file and a short pointer at the top of each of
`docs/reliability-learning-log.md` and `docs/practical-component-scanning.md`.
All 23 local links and 10 listed evidence directories were checked to exist;
the repository whitespace check passed. Historical result files and sealed
evidence were not rewritten, and no new experiment was left running.
