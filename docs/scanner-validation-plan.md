# MINI thorough validation plan — 2026-09-07

Requested by Ben: draw up a detailed plan and execute it, including firmware
flashing if justified. Work directly in this repository. No live experimental
meshing, unrelated service changes, commits, pushes or deployments.

## Objective and decisions

Find the fastest **useful** scanning configuration for this physical MINI,
not the setting that merely returns the most buffers. Separate five limits:

1. Depth acquisition rate and exposure-dependent completeness/repeatability.
2. Host preparation, tracking and display throughput/latency.
3. Motion during the structured-light measurement, versus tracking between frames.
4. USB/lifecycle reliability and long-run thermal behavior.
5. Firmware constraints that can be changed with a justified, recoverable update.

Deliver measured operating envelopes: detail-first, balanced, fast/compromised,
and unqualified. Record uncertainty and withheld claims. Do not infer handheld
mm/s from turntable success or claim accuracy from a tracker accepting frames.

## Fixed safeguards and experiment discipline

- Resolve serial `A22704522A6F00A64`, USB `2207:110c`, before every hardware run.
  Depth/RGB node numbers and USB addresses can change. One scanner owner only.
- Baseline currently verified: firmware `v2.8.7.20220610`, algorithm
  `v7.5.7.0426`, 640×400 Y16, frame/exposure 20,000/20,000 µs, auto-exposure 0.
  Confirm fresh state rather than assuming a previous successful restoration.
- Preserve full-resolution raw input. Short, duplicate and sequence-missing
  frames are separate metrics. Never synthesize frames or relax tracking gates.
- Change one factor at a time; repeat promising results in at least three blocks
  and bracket quality blocks with the original baseline. Keep physical view,
  distance and lighting fixed within each stationary block.
- Report median/p95/p99 frame gaps, complete/unique fps, burst pairing, stalls,
  raw versus accepted versus displayed frame rates, and end-to-end latency
  where timestamps actually support it. Delivery time is not exposure time.
- Record scanner CPU/NPU temperatures every five seconds; notify at 75°C,
  stop at 80°C or missing/implausible telemetry. Leave firmware thermal
  protections intact. These sensors do not measure the projector or optics.
- Each run has explicit frame/time/storage bounds and a restoration `finally`.
  Restore exposure/window in safe order using fresh device readback, reacquiring
  the serial-resolved node after closure. If the device disappears, stop issuing
  commands; do not loop resets or pretend restoration succeeded.
- Prefer one continuous stream for timing A/B tests; earlier repeated stream
  transitions caused USB enumeration failures. Reliability testing below is
  deliberately bounded, not a blind reconnect stress loop.
- Record host load and own-process CPU/memory when benchmarking. Do not stop,
  renice or otherwise change unrelated workloads. Label shared-load results.
- Keep scanner evidence under a fresh `/home/ben/scans/diagnostics/` run root,
  with machine-readable reports, raw-file hashes and exact settings. Cap initial
  raw storage at 15 GB; expand only when justified. Keep small durable summaries
  in this repository. Do not overwrite prior experimental evidence.

## Provisional quality acceptance criteria

These are engineering comparison gates, **not the vendor's metrology warranty**.
They are fixed before the new quality sweep, and may not be silently relaxed
to make a faster setting pass.

Compare the same baseline-visible surface pixels, with separate object ROI,
interior, edges/thin features and support-surface reporting. Largest-connected
depth region alone is not semantic object segmentation. Preserve loss maps.

- At least 95% of baseline mean return coverage overall and on edges; at least
  98% of baseline mean interior coverage.
- At most 2% of baseline-stable pixels become persistently missing. Stable:
  present in ≥90% of bracketing baseline frames. Persistently missing: present
  in <10% of candidate frames.
- No repeatable disappearance of a baseline-resolved target feature.
- At mutually observed stable pixels, p90 absolute median-depth shift no more
  than max(0.10 nominal mm, 2× the baseline-to-baseline p90 shift).
- Temporal MAD no more than 1.5× the baseline at both median and p90.
- A moved/drifting reference, empty ROI, insufficient returns, duplicate timing
  stamps or missing thermal/restoration record makes a block inconclusive,
  not a passing result. Report raw metrics even for failed settings.

Nominal depth units must be distinguished from verified physical dimensions.
An STL comparison additionally includes printing tolerance and registration
uncertainty. Check the print record and part identity before calling it ground truth.

## Stages and execution gates

### P0 — Baseline, instrumentation and preserved evidence

1. Inventory exact device, firmware/controller hashes, USB descriptors/speed,
   video format, runtime timing, temperatures and host resources.
2. Read existing print/CAD provenance and earlier firmware/recovery evidence.
3. Test the experiment harness offline: parameter bounds, fresh getters,
   partial-read/error/termination cleanup, restoration, sample labeling,
   timestamp validation and no hardware access from tests.
4. Record a fresh 20 ms baseline (128 complete frames) and confirm its rate,
   uniqueness, depth distribution, motion state and restoration.

Gate: proceed only with consistent identity/state, readable temperatures and
usable depth. Ask the operator to stop the turntable for P1; meanwhile perform
read-only firmware analysis and motion-independent host tests.

### P1 — Locate the exposure/quality knee

Run stationary A/B/A blocks with 40 settled complete frames per setting,
discarding ≥12 complete frames and ≥1 s after each transition. Initial ladder:
20 → 15 → 12 → 10 → 20 ms. Interpolated 12 ms lies within recovered 7–20 ms
presets, but still requires offline validation and fresh hardware readback.

Refine only around the pass/fail boundary (e.g. 18, 16, 14, 13 or 11 ms), not
an indiscriminate low-exposure sweep. Repeat finalist(s) in three interleaved
baseline/candidate/baseline blocks. The previous 7–10 ms failures remain useful
negative controls; repeat at most one as needed, not every failed setting.

Outputs: fixed-ROI return maps, interior/edge retention, persistent holes,
median signed bias, absolute bias p50/p90/p99, temporal MAD and outlier tails,
cadence distributions, baseline drift and confidence across blocks. Pixel
counts are correlated; do not pretend millions of pixels are independent trials.

Then decouple frame window from exposure using only valid exposure≤window
pairs: hold the window at 20 ms while changing exposure, and compare to matched
window/exposure. This distinguishes brightness/integration loss from any extra
effect of shortening the pattern timing. No exposure longer than its window.

Manual extensions: repeat finalists at two additional object orientations and
working distances; repeat under controlled ambient illumination. Do not move
the object between the A and B portions of a block. Record any test that needs
operator placement as pending until that placement is actually available.

### P2 — Host bottleneck and GPU ablation

Replay identical saved raw frames with CPU normals versus production GPU
normals, asynchronous versus synchronous coverage, and preview components
enabled/disabled **in the benchmark only**. Randomize A/B order, warm imports/
GPU kernels, measure ≥3 repetitions, and compare complete decisions and poses.

Instrument preparation, normal-neighbor gathering/covariance/eigensolve,
correspondences, solver, reference selection/overlap, coverage serialization,
depth/RGB/model display, disk writes, garbage collection and queue residence.
Record own-process/thread CPU time alongside wall time to expose contention.
Check memory/latency growth over long scans, not just a short warmed replay.

Only optimize stages that measurements justify. Preserve filtering policy,
sample count, arithmetic/gates or explicitly qualify a changed algorithm.
Validate every implementation change offline before another hardware test.

### P3 — Moving acquisition and geometric tracking

Use the operator's repeatable turntable speed, then low/medium/max settings
when available. Compare 20 ms, the P1 detail-preserving finalist and one faster
compromised setting. At least 300 complete frames or three rotations per run;
repeat each finalist three times with the same scene and physical speed.

Measure accepted/offered frames, retries/coarse recovery, sequence loss,
pose jumps, agreement/residuals, depth completeness and repeat-view consistency.
Do not stop at “all frames accepted”: compare image-matched revisits and shape
against the qualified reference, and inspect thin features for shear/smearing.
Use the existing target gate/CAD where print identity and unit assumptions permit.
Offline reconstruction is permitted; experimental **live** meshing is not.

Replay temporal strides 1/2/3/4 to isolate the tracker's displacement tolerance.
Label this as skipped-view stress: it does not reproduce motion corruption
inside one structured-light acquisition and is not a measured handheld speed.

### P4 — Real handheld speed envelope (operator needed)

Use a measured straight path and timed traversals: approximately 2, 5, 10, 20
and 40 mm/s, stopping escalation at repeatable quality/tracking failure. Three
passes per level in each direction. Add controlled arcs/rotation separately;
keep working distance and object visibility as constant as practical.

Use ruler/video/physical timing for independent motion ground truth. Tracker
poses alone cannot validate the tracker's own speed accuracy. Repeat successful
levels with the P1 finalist and baseline. Define the usable maximum as the
highest repeatably passing level with a margin below the first failure, not
the fastest single lucky pass. Low-feature/reflective objects are separate cases.

### P5 — Thermal endurance and USB lifecycle

For a quality-qualified candidate: 5 min shakedown, then 15 min, then 30 min
only if the previous stage is stable. Save bounded raw excerpts each minute
for thermal drift/quality comparison; keep complete timing/hash/telemetry logs.
Repeat baseline/candidate quality checks warm and after cooldown. Measure the
last-five-minute temperature slope; a short run is not an indefinite plateau.

Run a separate raw maximum-tested-rate soak only if needed to distinguish
embedded acquisition from GUI load (the earlier 7 ms 5 min/12.99 fps result is
already evidence). Apply the same telemetry/cutoff policy throughout. The
executed 15-minute extension tests continued warming and cadence stability,
not P1 quality: 7 ms is still unqualified. Do not automatically promote this
acquisition-only success to candidate endurance or a 30-minute scan.

After successful steady operation, perform at most five normal start/stop
cycles, with fresh state and descriptor checks between them. Abort on the first
unexplained USB fault. Test timing changes while streaming versus across a
closed stream separately; preserve host/device logs rather than mixing causes.
Do not use USB resets, power cycling or firmware changes as routine cleanup.

### P6 — Firmware and SDK investigation, then gated intervention

This stage starts with offline/read-only work alongside earlier preparation.

1. Compare recovered controller and SDK code for sensor modes, gain/exposure,
   frame timing, pattern count, depth confidence and output scheduling. Bind
   command IDs to exact-unit implementation evidence; generic SDK names are
   not proof that a command exists on this firmware.
2. Identify board/boot/update layout, installed version and factory calibration
   locations from bounded reads. Recover and hash relevant existing artifacts.
   A controller ELF or calibration JSON is **not a full firmware backup**.
3. Check official MINI (not MINI 2/POP) update availability, release notes,
   compatibility, supported updater platform and documented recovery. Inspect
   official packages offline where available. Do not invent image URLs or use
   an unidentified forum binary as an update candidate.
4. Prefer a validated reversible runtime setting or RAM-only experiment if it
   answers the hypothesis. Gain requires a recovered setter/range/readback;
   projector power, thermal protections and calibration are not tuning knobs.
5. Flash only if there is an exact-compatible image, authenticated provenance/
   integrity evidence, a supported update path, preserved device-specific
   calibration, a viable rollback/recovery route, stable power, and physical
   access if recovery needs a button/cable. Name the expected improvement and
   record the pre-flash baseline. The user's authorization includes a justified
   flash; it does not make an unidentifiable image or unavailable recovery safe.
6. After a justified flash, independently verify identity/version/calibration,
   rerun P0 and P1 before motion/endurance, and roll back if quality or reliability
   regresses. Do not change host algorithms during the firmware A/B comparison.

If the image, preservation or recovery gate cannot be satisfied, stop the flash
branch and report the exact missing prerequisite. Continue the other in-scope
tests rather than declaring the whole investigation blocked.

## Initial source references

- [Official MINI specification](https://global.revopoint3d.com/en-hk/products/official-refurbished-mini): up to 10 fps is the advertised rate, not proof of a firmware cap.
- [Official Revo Scan 5 manual](https://revopedia.revopoint3d.com/en/Revoscan/Revoscan5usermanual): device-specific firmware update UI.
- [Official MINI quick-start guide](https://download.revopoint3d.com/support/download/mini/mini-quickstartguide-en-v3.0-2023.7.27.pdf): consult exact-model recovery instructions before relying on them.
- [Previous measured timing/firmware investigation](scanner-fps-investigation.md).
- [Previous stationary quality and live GPU evidence](live-gpu-throughput.md).

## Execution ledger

Detailed observations, completed/pending stages and artifact paths are tracked
in [scanner-validation-results.md](scanner-validation-results.md). A written
test is not a completed test. Operator-dependent stages and unmet firmware
gates remain explicit; do not imply that the complete matrix ran unattended.

## Evidence-driven extension, 07:40 UTC

The original baseline-retention gates remain unchanged. New stationary data
showed repeatable exposure-dependent spatial rippling on three CAD-planar faces:
20 ms / gain-register 24 produces approximately 0.15–0.22 nominal mm median-map
plane RMS, versus 0.02–0.05 mm at shorter exposures. Returning to baseline
restores the ripple. Reducing gain to 16 at 20 ms also greatly reduces it.
Therefore, agreement with the original baseline is not an accuracy oracle.

Add a **separate, explicitly labeled clean-reference comparison**: retain the
original 20 ms / gain 24 at the start and end, but bracket candidate 12 ms /
gain 32 with repeated 20 ms / gain 16 references inside the same stream. The
same numerical retention/repeatability gates apply relative to that reference;
do not erase the failures against the original baseline. The comparison only
qualifies the current stationary view, not absolute dimensions or handheld use.

The exact controller's gain setter/range and fresh getter have now been
recovered and tested (see results). Closed gain candidates are only 16, 24 and
32; no projector-power change. Restore original gain as well as timing under
the existing termination shield. Allow up to ten stages, still 800 offered
frames / 150 seconds maximum, to accommodate three internally bracketed
candidate repetitions in one continuous stream. Preserve gain labels throughout
analysis. Capture sizes remain well within the 15 GB initial storage bound.

## Second-view refinement, 2026-09-07 16:00 UTC

After the operator rotated and stopped the target, three 15 ms / gain 24
repeats measured ~6.14 fps. Relative to the separately declared 20 ms / gain 16
reference, edge retention was ~96.6%, p90 shift was at the 0.10 mm boundary,
and median temporal MAD was 1.538× the mean of the two bracketing references,
just over the unchanged 1.5× limit. This is not a passing candidate.

Refine at 14 and 13 ms with gain held at 24, each separated by 20 ms / gain 16,
and retain original 20 ms / gain 24 at both ends. This tests the intensity-
artifact trade-off near the boundary, not an assumption that shorter exposure
must be better. Preserve both original- and clean-reference reports. If a
candidate is promising, require three interleaved repeats before advancing.
The full-resolution map shows the top of the rotated target touching the
image boundary; results cover the observed view, not the complete part.

Fixed planar patches were selected from the baseline image before computing
their plane metrics. Candidate planes are fitted only to measure flatness;
matched-pixel shifts are not registered away. A cropped connected-region
comparison (`x=140..475, y=10..277`) excludes surrounding returns but is still
not claimed as exact semantic segmentation or complete thin-feature metrology.

## Front-side revisit and operator contamination warning, 16:22 UTC

The next operator rotation appears to revisit the original front side in a
slightly different position, not supply a third distinct view. Rotation
direction is unknown and must not be invented. A 14 ms / gain 24 three-repeat
block on this view is still useful because that setting was not previously
qualified here.

The operator reported that an arm entered a scan; exact frame timing is
unknown. Preserve and flag both `p1-orientation-check-03` and
`p1-front-revisit-14ms`, withholding qualification. A completed/restored stream
does not override this observation. Analysis must respect the validity
annotation; do not silently crop the warning away or remove inconvenient frames.

Check individual raw frames against a fresh baseline reference, including
added surfaces and lost/shifted reference depth. Scene-change thresholds are
inspection aids, not semantic arm detection or replacement acceptance gates.
Median maps alone can conceal transient occlusion. Acquire a new, visually
unobstructed 32-frame view check, then a separately named replacement block
with the same three 14 ms / gain 24 candidates and reference stages. Preserve
the original- and clean-reference outcomes separately, with unchanged gates.

## Third distinct stationary view, 16:42 UTC

The operator rotated and stopped the part after the half-turn request. The
32-frame check `p1-orientation-check-05` shows a distinct back/other-side view,
stationary and without an apparent occluder. No exact physical angle is known.
The main part is not clipped, but its top has only two pixels of margin.
The largest connected baseline region extends onto the turntable/support.

Before the quality block, record fixed object-area rectangle
`x=110..474, y=0..277` and four planar inspection patches in
`p1-third-view-rois.json`. Compare the unchanged three-repeat 14 ms / gain 24
block against both original and clean references, for the full connected
region and this fixed crop. Report both; a favorable cropped result cannot
erase a full-region failure or establish exact semantic/whole-part coverage.
Retain full-frame maps, scene-change audit and poor planar patches. No
candidate pose registration, thermal-policy change or production default change.

The third-view 14 ms / gain 24 three-repeat block meets the clean-reference
numerical gates in both the full region and the fixed crop, matching the prior
front/side results. Advance this **provisional stationary finalist** to the
planned five-minute acquisition shakedown at exactly 14/14 ms, gain 24, with
original 20/20 ms restored afterward. This does not promote it to a general
tracking, distance/lighting or complete-part accuracy default. Original-reference
failures remain recorded independently.

The existing soak harness now has a closed 7/14 ms selector (default remains
7 ms), fresh gain checks, and requires a predecessor at the same timing for
longer stages. Keep the 80°C cutoff, five-second telemetry and bounded ten raw
excerpts per minute. Run only the 300-second stage here, followed immediately
by the unchanged nine-stage 14 ms quality block in this same fixed view. No
automatic 15/30-minute escalation; report warm-versus-initial shifts separately
from each block's internal repeatability.

## Post-replug lifecycle isolation, 17:23 UTC

Fresh inventory verifies the original timing/gain, identity, firmware and
depth-intrinsics hash after physical reconnection. The previous five-minute
failure remains a failed lifecycle outcome, not a successful predecessor.

Predeclare at most five short, instrumented 64-frame capture cycles:
`baseline`, `post`, `live`, `post`, `live`. Baseline never changes timing;
post configures 14 ms before acquisition and restores 20 ms after stop;
live restores 20 ms at the first five-second monitor after at least 32
dequeues, while acquisition is still active, then ends at 64 frames. Thus
the live variant deliberately includes baseline tail frames; its mixed rate
is not a single-setting throughput result.

Record each UVC request's selector/query, timing, phase, outbound bytes and
inbound hash/count, preserving the first transport failure before cleanup.
After ENODEV/EPROTO/timeout/I/O/pipe transport errors, issue no further UVC
requests; do not mask the original fault with a cleanup retry. Stop the
matrix at the first fault. Preserve required timing restoration for non-
transport exceptions while the serial-bound device remains accessible.
Read chip temperatures every five seconds with the unchanged 80°C cutoff.
Read bounded memory/uptime files before and after each cycle. No controller,
driver, firmware, power or algorithm change. All raw frames/reports remain
separate from quality-qualified blocks, and logging can perturb timing.

### Autosuspend hypothesis branch, after baseline lifecycle 01

The first baseline cycle captured 64 complete unique frames and verified
20/20 ms, auto 0, gain 24, but the host recorded USB disconnect ~0.22 seconds
after the report ended. It automatically returned at the next device address.
The existing stream code already documents brief stop-associated re-enumeration;
this is distinct from the preceding five-minute run's failed recovery. Abort
the remaining original matrix: the old report missed post-close topology.

Read-only host inspection finds `power/control=auto`, a 2000 ms autosuspend
delay, and the scanner runtime-suspended. This is a hypothesis, not a root-cause
finding. The [Linux USB power-management documentation](https://docs.kernel.org/driver-api/usb/power-management.html)
explains that an open usbfs file prevents idle autosuspend. Test exactly one
additional baseline lifecycle with a read-only usbfs descriptor held across
capture, readbacks, and all V4L2 closes. Resolve exact serial/USB IDs and verify
the opened character-device number. Issue no USB ioctl, claim, detach, reset,
sysfs/udev/global power write, or firmware change.

Record cached sysfs topology/power state at control phases and every 50 ms
for four seconds while held after the final V4L2 close, then release the
descriptor and observe six more seconds. Close it even on interruption/error.
Any disappearance/device-number change makes the lifecycle unstable, separately
from successful timing restoration, and prohibits subsequent device operations
within that run. A successful held phase would justify bounded confirmation;
a fault while held would argue against autosuspend as the sole explanation.

### Quiet-stop / gadget teardown branch, 17:44 UTC

The held baseline also re-enumerates while its old device remains active with
runtime usage 1 and unchanged suspended-time counter. Therefore blocking host
autosuspend is not a sufficient fix. Fresh no-stream inspection succeeds and
shows continuous device uptime. The recovered controller's bulk timeout path
has a one-second select timeout, stops the DSP, stops streaming, and signals
the gadget manager. Its manager closes/recreates gadget threads; this is a
candidate explanation for delayed USB disappearance, not proof of the previous
unrecoverable failure's cause. No binary patch is justified from strings alone.

Run one baseline 64-frame `--quiet-stop` case with no timing writes, no UVC
operations after capture closes, and bounded sysfs observation. Preserve any
topology change as unstable rather than calling it a clean lifecycle. After
automatic return is stably observed, use a separately labeled no-stream inspect
to capture the controller log before verbose getters, then fresh state/gain,
temperature, uptime and kernel evidence. No automatic reset/retry. If this
confirms the ordinary timeout/restart sequence, a bounded live-restoration
case may test moving required restore/readback work before shutdown; it still
requires separate post-return verification and does not cure the firmware.

The quiet baseline's controller log confirms `bulk_timeout exit`, depth-stop,
shutdown of both camera gadget threads, and reopening of the same DepthCam/RGBCam.
The first live-restore/quiet-stop test and a separately labeled post-return
inspection both verify original timing/gain without a UVC error. Its topology
change remains explicitly failed under the earlier no-change criterion.

Now test one `settled-post` cycle: 64 frames at 14 ms, six seconds with **no
UVC requests** after capture, then require the exact identity stable throughout
the last two seconds before reopening a fresh descriptor. Check 14 ms/gain 24,
restore 20 ms, and verify timing/gain/temperature. This explicitly accepts the
observed stop-associated, automatically recovered re-enumeration as a boundary;
retain the old USB epoch separately. It does not accept control errors, a
missing/unstable device, or any reset/retry to make the test pass. This boundary
is a host workaround candidate, not firmware repair. Only after it passes may
the unchanged five-minute acquisition stage be retried with this quiet boundary.

`p5-lifecycle-08-settled-post` passed this separately defined recovery boundary:
64 complete unique frames, no UVC error, 14 ms/gain 24 freshly read after device
92→93 re-enumeration, 20 ms/gain 24 restored, and stable final USB observation.
The thermal harness's opt-in `--quiet-after-stop` now applies the identical
six-second silence/two-second identity requirement, records UVC and STREAMOFF
boundaries, and prohibits control retries after a transport/settling failure.
It is limited to 300 seconds; no automatic longer-stage qualification.

Repeat the unchanged nine-stage stationary block as `p5-retry-initial-quality`,
then one 300-second 14 ms acquisition with the quiet boundary. Keep original
and clean comparisons, the existing fixed object-area ROI, and no candidate
registration. On success immediately acquire `p5-retry-warm-quality`, then
compare each stage to the fresh initial stage at fixed initial-reference pixels.
The delay through shutdown/readback and the warm block's stage/temperature
history remain explicit; this is not an instantaneous or causal thermal test.

The retry completes with 1,975 unique complete frames, verified restoration,
and a stable post-readback USB epoch. The warm block also meets its internal
clean-reference numerical gates, but fixed initial-pixel p90 shifts at 14 ms
are 0.160–0.173 mm; clean 20 ms references shift 0.150–0.165 mm too. These
cross-block changes are retained, not aligned away or declared unique to the
faster timing. Do not escalate to 15/30 minutes or claim temperature-invariant
accuracy on this evidence.

After idle cooling, the 18:12 UTC no-stream check reads ~59.4/59.1°C, versus
~62.5–63.8°C CPU during the warm block. Acquire one identically labeled
nine-stage `p5-retry-cooldown-quality` block, with the same ROI and original/
clean references. Compare fixed initial pixels and all stage temperatures;
cooling is not perfectly matched to the initial thermal history, and partial
return of depth values would support a reversible component, not prove that
fixture motion or optical/calibration drift is absent. No new warm-up correction.

The cooler block completed/restored, with all three internal clean-reference
numerical comparisons passing. Fixed-initial-pixel candidate p90 shifts fall
to 0.0925–0.0950 mm, versus 0.1600–0.1725 mm warm; clean 20 ms references
also partly return (0.0850–0.0950 mm versus 0.1500–0.1650 mm). This is one
partially reversible heat/cool sequence, not temperature-invariant accuracy
qualification or proof of the physical drift mechanism. Preserve original-
reference failures and the tighter margins in earlier views. The bounded
replug/lifecycle/five-minute retry branch is complete; longer candidate soaks
remain unqualified pending fixture/thermal repeatability checks. Next motion
work requires the operator to start a fixed repeatable turntable speed; do not
infer actual motion, rotation period or handheld mm/s from the current stopped
scene. Full results and current restored state are in the execution record.

## Fresh low-speed motion branch, after operator start at 18:39 UTC

The operator started the turntable at the requested lowest steady speed and
warned that accessing its underside may have shifted its placement. Treat
this as a **new physical setup**, not a registered continuation of the old
stationary view. Do not transfer old pixel masks, axis/centre fits or pose
ground truth. Record fresh framing, usable depth band and image-revisit timing.
Image-derived periods are not independently measured motor speed.

Preserve new evidence in `/home/ben/scans/diagnostics/20260907-motion-low/`.
First use the already tested 64-frame 14 ms settled-post lifecycle case as
a moving framing check, with no tracking or meshing and the six-second quiet
boundary. Inspect full frames; if the moving part is clipped or obscured, ask
for repositioning rather than qualify that data as complete-part scanning.

If framing permits, capture bounded acquisition-only motion blocks with full
raw frames, fresh timing/gain verification, five-second thermal telemetry and
the same quiet-stop/first-fault protection. Start with 300 complete baseline
20 ms frames, then 300 at the 14 ms finalist. Require sufficient credible
revisits before calling either a rotation-consistency test. Retain all startup,
settling, short and sequence-loss metadata, while excluding declared settling
frames from subsequent comparisons. Do not compare fixed image pixels across
different rotation phases as if stationary.

Replay with the unchanged production GPU-assisted tracker and fixed common
preparation settings. Evaluate acceptance **and** actual individually matched
revisit geometry, not tracker-derived speed alone. These replays separate
acquisition/motion quality from host backpressure; they are not live-GUI fps.
If the first pair yields interpretable results, repeat baseline/finalist in
reverse order, then complete a third pair. A compromised 10 ms comparison is
optional diagnostic follow-up, not a newly quality-qualified mode. Keep each
capture at most 180 seconds/1,200 delivered frames and this first low-speed
branch below 15 GB. Stop on a transport/thermal/scene fault; no automatic
reset, driver/power changes, new tracking gates or firmware intervention.

The 64-frame placement check appeared usable with two pixels of top margin,
but the longer 320-frame original baseline reaches the top image row in
98 frames. Full images confirm the tallest target feature is clipped during
part of the rotation. Preserve this as acquisition/lifecycle evidence only;
its validity annotation blocks quality qualification. Pause further motion
captures and the candidate comparison until the operator reframes the target
with top margin throughout rotation. The raw-capture/replay instrumentation
is prepared and mock-tested, but no GPU replay of this invalid scene or
physical-speed result is claimed. Restart baseline/candidate evidence in a
fresh setup after adjustment, at the same turntable setting if possible.

## Reframed continuation, 19:02 UTC

The operator adjusted the scanner and requested continuation. No interrupted
scanner process still owns a video node. Start a new evidence root,
`/home/ben/scans/diagnostics/20260907-motion-reframed/`, with a 320-delivery
20 ms baseline and full moving-framing inspection. The prior clipped capture
remains excluded. If the new view has usable margins, proceed with the existing
baseline/14 ms motion comparison and unchanged GPU replay, using an explicit
common 60..300 nominal-mm preparation band unless fresh framing/depth evidence
requires a separately declared change. Extend to the already closed 640/1200
delivery budgets only if the shorter blocks lack credible image revisits; keep
the 180-second per-capture ceiling, thermal monitoring and quiet-stop boundary.

The reframed 320-frame original baseline has no top contact (minimum 52 pixels),
but connected returns reach the bottom in 133 frames. Inspection must distinguish
the target base from the original setting's stronger support/clutter returns;
connected-region contact alone is not semantic clipping. Take one 320-frame
14 ms block as an additional **framing diagnostic**, not an already qualified
motion comparison. Inspect its full outline before advancing. The first 300
settled baseline frames span 66.36 seconds and have no credible image period;
no rotation-consistency or motor-speed claim follows from that short prefix.

The 14 ms diagnostic confirms that the base has insufficient lower margin,
with solid base corners meeting the bottom in selected full-frame views.
Neither capture qualifies as whole-part motion evidence. Bottom-connected
counts include support and must not be described as exact clipped-target counts.
An already executed 300-frame baseline GPU replay accepted every offered frame
but found no credible image revisit; retain it as an unqualified diagnostic,
not a geometry pass. No candidate replay or longer capture in this setup.

## Recentered continuation, 19:17 UTC

The operator opened the app to inspect the camera and moved the placement to
clear the bottom edge. No video node has an open owner at the subsequent check.
Use a third, separate root, `/home/ben/scans/diagnostics/20260907-motion-centered/`.
Do not merge either earlier clipped setup with this placement. The initial
300-frame baseline prefix was too short to obtain a credible image revisit,
so start with the existing 640-delivery 20 ms baseline (180-second ceiling).
Inspect full-frame margins over the entire capture before candidate work.
If usable, acquire the 14 ms candidate with the existing 1,200-delivery ceiling
and 180-second time limit; report actual delivered counts, not requested counts.
Replay settled complete inputs using the unchanged GPU tracker and common
60..300 nominal-mm depth band. Revisit checkability and geometric agreement
remain separate from frame acceptance. No meshing, firmware change, physical
RPM claim or increased turntable speed. All previous thermal/lifecycle limits
and the reverse-order/three-pair qualification plan remain in force.

### Limited diagnostic branch, declared before centered replay

The 640-frame baseline has at least 50 pixels of top space, but a few base
corners reach the lower image boundary in selected orientations. Support
returns also contact that edge. Thus whole-part margin/coverage is still not
qualified; no exact amount of lost target depth is established. Preserve that
warning rather than silently passing the framing gate.

The major target features remain visible, allowing a narrower, useful question:
does the unchanged tracker show gross inconsistency when the visible geometry
repeats? Permit **diagnostic-only** replay of the flagged baseline, then one
bounded 14 ms comparison if the baseline is interpretable. This is not the
three-pair full-part qualification matrix. The explicit `--diagnostic-scene`
option carries the unchanged source warning and `quality_qualification_allowed:
false` into replay and independent audit reports, including per-stride files.
It does not relax capture success/restoration, hashes, settling, timing,
preparation, registration or geometric agreement gates. Ordinary replay still
refuses a flagged scene. Test this reporting path before physical-data replay.
Any geometric success remains limited to observed surfaces; missing coverage,
unobserved views, physical speed and full-part quality remain unqualified.

The first centered baseline replay is interpretable: 628/628 accepted and
credible image matches roughly one rotation apart, with individually matched
agreement above the unchanged 75% drift gate. Proceed with the declared single
14 ms diagnostic capture. Also replay the same saved inputs at strides 2, 3
and 4, first baseline then candidate, to isolate tracking across larger
between-frame displacement. These are not faster physical acquisition or
handheld-speed simulations: motion within the scanner's pattern sequence is
unchanged. Retain all stride results, including refusal/loss and uncheckable
revisits. Shared-load processing times are not isolated live-GUI ceilings.

The candidate's observed-geometry replay fails both legacy and independently
matched drift checks despite accepting all frames. Its longer duration tests
additional revisit angles, so do not attribute the difference to exposure
alone. Add one **20 ms, 1,200-delivery / 180-second** diagnostic control in the
unchanged placement, to broaden baseline revisit coverage to the same time
ceiling. Report actual counts. Separately inspect the candidate's worst actual
image pair with direct two-frame alignment and compare its view to the baseline
raw images. This fit is a diagnostic contrast, never a correction substituted
into the original trajectory or a held-out validation result. All warnings,
failed checks, thermal/restoration constraints and scope limitations remain.

Direct inspection of the worst stride-1 candidate pair (source indices
195/948) finds matching raw geometry: about 99.7% projection agreement under
identity or direct two-frame alignment, versus 64.6% under accumulated poses.
The initial short baseline's closest matching view is source index 280, whose
next ~519-frame revisit lies outside that capture. Its good score therefore
did not test this same view twice. The longer baseline control is in progress.

Before interpreting its different starting phase as an exposure effect, replay
the **same candidate bytes** starting at source index 195, retaining all 976
remaining frames, without reordering or changing timestamps. This is a
diagnostic starting-view experiment, not a new acquisition. The explicit
`--start-offset` is allowed only with diagnostic-scene mode, is recorded
separately from the original 12-frame settling exclusion, and preserves hash,
time and source-index provenance. Keep the original failed trajectory intact.
Any improved result would show dependence on registration history; it would
not validate the whole original scan or repair its output. Test before replay.

The known pair improves 64.632%→99.714% when tracking starts there, although the
shifted window's automatic period sampler remains inconclusive. The longer
original-speed control passes its own sampled observed-view gate at 82.585%,
with all 797 frames accepted. Neither result erases the candidate's original
failed trajectory or qualifies full-part coverage. Close and preserve this
bounded diagnostic block. Next prioritize exact-byte CPU/GPU parity on the
new failure and isolated anchor/reference/revisit experiments before claiming
that more throughput fixes tracking. The original three-pair motion matrix,
physical-speed/handheld tests, full framing and thermal/firmware gates remain
pending; no production tracker change or overall-plan completion is asserted.

## Tracking reliability continuation, 20:00 UTC

The user requests continued testing and fixes toward consistent reliable
operation. Keep earlier diagnostic roots sealed. New evidence belongs in
`/home/ben/scans/diagnostics/20260907-tracking-reliability/`.

First replay the exact failing 1,171-frame candidate sequence with CPU reference
normals, then repeat GPU, leaving preparation, source order, policy and gates
unchanged. Compare complete decisions, poses and individually matched revisit
checks, not timing alone. The CPU path uses the explicit reference estimator
with the same bounded context-local cache and must never initialize GPU compute.
Verify the new selector offline before replay. Then isolate reference-retention
changes in diagnostic-only experiments; the failing candidate is development
data, not held-out validation. Require controls, the earlier difficult moving
recording and fresh physical scans before promoting a production change.
No quality/acceptance gate is relaxed to make an experiment pass. Preserve all
failures, first-fault/quiet-stop behavior, temperature limits and no-live-mesh
requirement. Full-plan and reliable-handheld claims remain gated by evidence.

At 20:12 UTC, the CPU reference reproduces all 1,171 original GPU decisions
and poses exactly, with identical raw inputs and coherence. Its independent
matched-revisit score remains 64.632%. A fresh GPU repeat is running.

Predeclare two closed, offline reference ablations on this development sequence:
`retention64` raises only keyframe/cloud retention from 12/32 to 64/96, leaving
the eight-reference/1,800-model-point/900-source-point work budgets unchanged;
`no-model` removes only the merged-model candidate, retaining adjacent and
keyframe checks and unchanged recovery/gates. Run separately, with unchanged
bytes, order, settling and framing caveats. Record the entire policy; non-default
profiles always forbid quality qualification, including on legacy recordings.
Do not change production defaults. Evaluate all sampled revisits and throughput,
not just the preidentified worst pair. Controls and earlier difficult recordings
are validation data, not a parameter-search grid. Any new experiment motivated
by these results must be declared separately and retain all failed outcomes.

The first retention64 replay accepts all 1,171 frames and passes the original
four credible actual-image revisit pairs at 95.183% worst agreement (legacy
fixed-lag samples: 99.163%). Its 7.864 processing fps excludes GUI/save/audit.
At source 948 it now uses first-pass keyframe 194, unlike the default trajectory.
This is development evidence only. The separately declared no-model ablation
is running. Validate retention64 unchanged on the two original-speed controls,
the earlier complete 2,703-frame moving recording, and fresh physical data.

Before promotion, add a denser independent 25-position image-matched revisit
audit using the existing image matcher and unchanged 75%/1.5-nominal-mm gate.
Hash-check every frame used in selecting pairs. Preserve the original sparse
and fixed-lag reports; an uncheckable dense estimate remains uncheckable.
This broadens angle sampling within recorded coverage, not full-part accuracy.

One isolated performance check is justified: 64 keyframes have over 4,000
directed pose pairs, exceeding the existing 2,048-entry exact-result cache.
Test a bounded 16,384-entry cache against the same retention64 sequence, requiring
exact complete-decision and pose equality, not merely similar quality scores.
This changes memoization only, not registration arithmetic or retention policy.

The cache-only replay is exactly identical in complete decisions/poses/policy
and improves measured processing from 7.864 to 9.960 fps. The denser audit has
22 credible pairs: retention64 worst 90.203%, original worst 58.321%. The
no-model experiment fails at 0% worst agreement and will not be promoted.

Fresh acquisition after the saved-recording controls: predeclare one bounded
14 ms/gain-24 **2,400-delivery / 360-second** motion capture, longer than the
previous 180-second clips so repeat views span more of the rotation. Actual
rotation is to be established from recorded images, not assumed motor speed.
The exact serial, ≥12-frame/≥1-second settling exclusion, original restoration,
five-second temperature readings, 80 C cutoff and first-fault/quiet-boundary
rules all remain unchanged. The 360-second option is closed to the new 2,400
frame budget; old 320/640/1,200 limits remain 180 seconds. Verify successful
and failed paths offline before any acquisition. Raw payloads remain bounded
to about 1.23 GB and saved after stream closure, including partial failure data.
This is a fresh motion test, not a 15/30-minute thermal qualification. Framing
must be inspected and any previous full-part coverage warning carried forward.

The older 2,703-frame recording remains a severe failure under retention64:
3.909% sparse and 0.981% dense worst agreement, versus original 11.9% sparse
and 6.4% dense. Do not promote retention64 to the app. The fresh six-minute
capture completed 2,371 unique complete frames at 6.598 fps without sequence
gaps/shorts, peak 64.2/63.5 C and verified quiet-boundary restoration.

Predeclare one additional reference-retention hypothesis before new replays:
`stable64` retains the same 64/96 budgets, but moves recency protection entirely
to the existing recent-cloud buffer. For eviction, evaluate unordered pose
pairs in a canonical direction and discard the newer member of the closest
pair (newest victim on exact distance ties), keeping the anchor. This addresses
continual replacement of previously useful references, without introducing a
turntable prior, first-revolution labels, image-matched loop constraints or
altered acceptance gates. Keep it in the offline diagnostic helper, record
the algorithm name as well as full policy, and test archive/recency invariants.
The old failing recording is now development data for this hypothesis; it is
not subsequently to be called untouched validation. Freeze this choice before
examining its results on the new six-minute recording and earlier controls.

For the remaining saved-byte geometry comparisons, independent replay processes
may run concurrently to reduce latency. They never share tracker state or raw
outputs and never access the scanner. Their recorded wall times are shared-load
diagnostics, **not throughput benchmarks**. The earlier isolated exact-cache
comparison remains the processing-speed evidence. The fresh framing audit has
no absent component and ≥50 pixels of top margin, but 272/2,371 frames touch a
border with the largest part/support component. Explicit diagnostic-only
qualification and the inherited base/support warning remain; do not trim them
away or call the whole part covered.

At 20:45 UTC, direct source/report inspection clarifies that the production
preference is actually `(model, adjacent, keyframe)`, despite a module overview
describing keyframe before adjacent. The no-model replay chose adjacent on
1,169/1,171 frames (one anchor, one coarse recovery), so its failure does not
test selecting a persistent keyframe. Add one closed `keyframe64` diagnostic:
same 64/96 retention and production eviction, all three candidates and their
unchanged gates/arbitration, but `(keyframe, model, adjacent)` preference inside
the agreed cluster. The chosen overlap also drives existing keyframe admission;
record those resulting decisions rather than claiming that effect is isolated.
Use a scoped process-local override only in the offline helper, restore it even
on exceptions, and record the complete preference for every profile. Freeze this
choice before examining its results; do not change production preferences.

The fresh retention64 trajectory exposes a **production consistency-check false
pass**: legacy fixed-lag samples pass at 84.203%, five actual-image pairs at
99.541%, but the 25-position actual-pair audit fails at 30.752%. Fix the shared
final consistency check to sample 25 image positions and project every credible
actual best pair, not substitute median-lag pairs. Keep 75%/1.5 unchanged,
preserve pair/source provenance and report unprojectable pairs as inconclusive
(never a clean pass). Add a synthetic regression where median-lag sampling
misses one demonstrably displaced revisit, then verify on the exact real
false-pass recording. This prevents false qualification; it is not a tracking
repair. Existing running replays loaded the old check, so their original and
new independent audit outputs must both remain identifiable.

Acquire one second 14 ms/gain-24 2,400-delivery/360-second confirmation capture
before choosing among the current profiles. Keep its poses unexamined until
the current comparisons are complete, so it remains a separate physical
confirmation rather than another parameter-selection sequence. Same temperature,
ownership, restoration, quiet-boundary, raw preservation and framing rules.

The keyframe64 experiment raises `NotARotation` before its first 100-frame
progress message. Independently, a **stationary mathematical pose** evaluated
as `p.compose(p.inverse().compose(p))` fails at iteration 20: its rotation
orthogonality error grows from ~3e-16 to >1e-6. This isolates a floating-point
coordinate round-trip feedback defect without scanner data or GPU computation.
Preserve that failing reproducer and the partial real replay. Fix only tiny
roundoff defects on composition: after the unchanged 1e-6 proper-rotation guard,
project defects above 1e-12 and at most 1e-10 back onto a proper rotation using
one polar correction. Larger defects/reflections remain subject to the original
refusals, not silently repaired. Verify 1,000 stationary round trips, normal
composition/inverse/registration regressions, CPU/GPU real prefix parity and
full real trajectories. This is a numerical correctness change, not a loosened
geometry gate or an assumed solution to the remaining registration drift.

The repaired keyframe-first trajectory still fails: 0% on the first fresh
six-minute recording, 2.416% on the older recording (2,702/2,703 accepted).
Reject it as a tracking mode while retaining the independently proved numerical
fix. Post-fix geometry/tracking/ICP/capture tests: 570 passed, 2 skipped.

Close this alignment-diagnostic block with two predeclared, separate ablations
on the development 1,171-frame candidate and first fresh 2,359-frame recording:
`untrimmed` changes only ICP keep-fraction from the tracker's actual default
0.8 to 1.0; `point-to-point` changes only the fitting objective to the existing
point-to-point path, with 0.8 keep-fraction and CPU compute (no GPU normals).
Keep default 12/32 retention, original candidate preference, preparation,
frame order and all acceptance/consistency gates. Record the actual fitting
method and fraction, not only the policy. These test local fitting/feature
discard as remaining drift mechanisms; no parameter search or default change
is authorized by a favorable single recording. The second physical capture
remains reserved for one final confirmation of the best supported choice.

At 21:33 UTC, both alignment ablations fail the unchanged dense consistency
gate on both recordings. No alternative retention, eviction, preference or
fitting profile is supported for promotion. Use the reserved second six-minute
capture for a single confirmation with **production default tracking policy**,
including the independently tested numerical roundoff correction, exact cache
enlargement and final actual-image consistency check. Do not search this
confirmation recording for a better profile. Its framing audit again has
no absent component and a 50-pixel minimum top margin, but 271/2,371 combined
part/support components touch a border. Carry the diagnostic-only full-part
coverage warning forward; inspect views 0400 and 1185 before replay. Keep the
same 12-frame settling exclusion, 60..300 nominal-mm band and all remaining
2,359 frames in order. A favorable observed-surface result does not erase the
earlier failures or qualify reliable handheld scanning.

The reserved confirmation fails at 73.5528% despite all 2,359 frames being
accepted. An exact-byte pair-level control reproduces that recorded-pose score
and measures 99.9469% for the same two views in unchanged camera coordinates;
no fitting, trajectory correction or confirmation-profile search is introduced.
Close this bounded comparison block and preserve its failures. Keep only the
independently proved correctness/cache changes in production. Do not mark the
larger scanner reliability plan complete or treat the final check as a newly
implemented live drift detector.

Next controlled motion protocol, requiring the operator at the scanner:

1. Confirm the app/device owner and exact serial, close any preview normally,
   verify the unchanged baseline and temperature readings. Ask the operator
   to stop the turntable for handheld tests; do not mix unmeasured part motion
   with unmeasured camera motion.
2. Record a stationary lead-in, one short lateral out-and-back movement while
   keeping the entire target visible, then a stationary return to the starting
   view. Mark each phase with monotonic timestamps. If distance/speed is not
   independently measured with a physical guide and timing, label it
   qualitative, not a millimetres-per-second limit.
3. Repeat baseline and 14 ms candidate on the same supported geometry and
   measured path, then test pitch/yaw separately. Preserve all raw frames,
   complete/refused tracking decisions, phase times, temperatures and device
   restoration. Do not ask the user to reach into an active acquisition.
4. Audit return-view agreement independently of the constraints used for
   registration, plus full-part geometry/coverage against the existing physical
   target evidence. Keep non-returning passes explicitly inconclusive under
   the image-repeat check. A silent accepted-frame drift is a failure, not a
   reason to weaken acceptance or consistency requirements.
5. Only after a registration change generalizes across the archived adverse
   cases and fresh controlled motion should it proceed to candidate-rate
   15/30-minute thermal/depth-drift and sustained GUI tests. Those gates are
   still required; the two new six-minute acquisitions do not replace them.
