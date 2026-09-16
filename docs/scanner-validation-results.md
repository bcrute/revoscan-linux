# MINI thorough validation — execution record

Started 2026-09-07 UTC. See [the test plan](scanner-validation-plan.md).

| Stage | Status | Evidence / next dependency |
| --- | --- | --- |
| P0 baseline/instrumentation | Completed | 128/128 complete unique baseline frames, verified identity/timing/thermal state |
| P1 stationary quality knee | Three views plus a fresh initial/warm/cooler sequence tested | 14 ms / gain 24 remains a ~6.6 fps clean-reference numerical finalist; original-reference failures and cross-block depth drift remain |
| P2 host/GPU ablation | Offline matrix and short live GUI checks completed | 24 randomized replays plus three optimized repeats; awake-display GUI applied all 299 depth previews at ~10.4 fps |
| P3 moving geometry | Fresh low-speed diagnostics expose starting-view/history dependence; full qualification pending | 14 ms delivers 6.598 fps without acquisition loss but its tracked revisit scores only 64.6%; the same pair scores 99.7% when tracking starts there; whole-part coverage and repeated matched-phase comparisons remain unqualified |
| P4 handheld speed | Operator required | Physical, independently timed motion is not automated |
| P5 thermal/lifecycle | Failed first exit preserved; five-minute quiet-boundary retry passed | 1,975 unique complete frames at 6.598 fps, restored state and stable returned USB identity; warm shift partly reverses on cooling; no 15/30-minute candidate pass |
| P6 firmware investigation | Runtime stop/restart path confirmed; flashing gated | Exact-unit disassembly plus controller logs explain routine delayed USB restart, not the earlier unrecoverable failure; compatible image/recovery prerequisites still missing |

No new firmware flash or persistent device modification has been performed.

## Artifacts and baseline

Root: `/home/ben/scans/diagnostics/20260907-thorough/`. Full raw data stays
outside the repository. No captured file was overwritten. Settings and readback
are in each capture's `report.json`; raw frames are retained per labeled stage.

`p0-baseline/`, 07:18:38–07:19:11 UTC:

- Exact serial `A22704522A6F00A64`, USB `2207:110c`, 480 Mb/s.
- 640×400 Y16, original 20,000/20,000 µs, auto exposure 0 before and after.
- 128/128 complete **unique** frames, zero short/sequence-missing frames,
  **4.4915 fps**. Raw code-return fraction 25.59% over the entire sensor.
- CPU 57.6→58.9°C, NPU 56.9→58.2°C; no setting writes in this baseline probe.
- Recovered controller 166,412 bytes, SHA256
  `decdc3ad4e34449cefee8543c790866cfc87dd9279a873c3abab124b9754b0f1`,
  identical to the earlier recovered binary.
- The known version tuple on UVC entity 3 failed through the bound V4L2 driver
  (`ENOENT`); no detach/retry was attempted. The installed version was later
  freshly verified by reading its known firmware-version files instead.
- Host has unrelated active workloads. No unrelated process was stopped or
  reniced. Shared-load results are not isolated-machine performance ceilings.

## P1: stationary exposure, timing and gain

Every stage retains 40 full-resolution complete frames after ≥12 discarded
frames and ≥1 second settling. All completed blocks verified restoration.
Short-stage fps varies with the phase of the scanner's alternating long/short
delivery gaps; it is not a precise sustained-rate measurement.

Completed blocks:

1. `p1-ladder-a`: matched 20→15→12→10→20 ms.
2. `p1-factor-isolation`: 20/20 → 20/15 → 15/15 → 20/12 → 12/12 →
   20/10 → 10/10 → 20/20 ms (frame window/exposure).
3. `p1-refinement-a`: matched 20→19→18→17→16→15→20 ms.
4. `p1-reverse-repeat`: matched 20→10→12→15→20 ms.
5. `p1-gain-isolation`: 20/24 → 20/16 → 15/16 → 15/24 → 12/24 →
   12/32 → 10/32 → 20/24 (matched timing ms / gain register).
6. `p1-clean-reference-repeats`: original 20/24 around four 20/16 clean
   references interleaved with three 12/32 candidates.

The initial three 15/12/10 ms comparisons reproduce a ~0.32 mm p90 absolute
matched-pixel shift from the original 20 ms reference. Baseline-to-baseline
p90 shifts are only 0.05–0.08 mm. Thus the candidates **fail the predeclared
original-baseline bias gate**; that finding is retained, not silently relaxed.
Holding frame timing at 20 ms while shortening exposure preserves the depth
shift, so it principally follows exposure, not just output cadence.

### Spatial shape changes the interpretation

Full-resolution maps in `p1-factor-spatial/`, `p1-reverse-spatial/` and
`p1-gain-spatial/` preserve median depth, return probability and temporal MAD
in `spatial.npz`. Plots use fixed scales and no candidate alignment. Difference
maps show periodic striping over visibly flat target faces.

Four fixed image patches are recorded in `p1-plane-rois.json`. Plane reports
are `p1-factor-planes.json`, `p1-reverse-planes.json`, `p1-gain-planes.json`.
The first three are clean, clearly visible planar faces; the broad base-front
patch remains less flat and is **not** discarded from the reports.

| Matched timing / gain | Ramp RMS | Left-face RMS | Right-face RMS |
| --- | ---: | ---: | ---: |
| 20 ms / 24, initial | 0.149 mm | 0.217 mm | 0.159 mm |
| 20 ms / 16 | 0.029 mm | 0.034 mm | 0.033 mm |
| 15 ms / 16 | 0.016 mm | 0.018 mm | 0.016 mm |
| 15 ms / 24 | 0.029 mm | 0.039 mm | 0.045 mm |
| 12 ms / 24 | 0.022 mm | 0.024 mm | 0.017 mm |
| 12 ms / 32 | 0.029 mm | 0.037 mm | 0.045 mm |
| 10 ms / 32 | 0.030 mm | 0.031 mm | 0.031 mm |
| 20 ms / 24, final | 0.150 mm | 0.210 mm | 0.159 mm |

These are plane RMS values of 40-frame median maps, **not single-frame
accuracy specifications**. No print-flatness calibration or complete CAD pose
fit is implied. The reverse-order exposure block reproduces the improvement.
The return of rippling on restoration, and its reduction with either exposure
or gain, supports an intensity-dependent artifact; saturation/nonlinearity is
a hypothesis, not yet verified from raw fringe intensities.

### Return preservation is a separate trade-off

Gain-isolation connected-region metrics (same mask within that block):

| Timing / gain | Short-stage raw fps | Interior returns | Edge returns | Persistent loss of stable pixels |
| --- | ---: | ---: | ---: | ---: |
| 20 ms / 24, bracket average | ~4.5 | 99.93% | 93.36% | 0% |
| 20 ms / 16 | 4.43 | 99.92% | 91.80% | 0.23% |
| 15 ms / 24 | 5.96 | 99.91% | 90.86% | 0.20% |
| 12 ms / 24 | 7.75 | 99.77% | 86.97% | 0.34% |
| 12 ms / 32 | 7.78 | 99.88% | 89.74% | 0.19% |
| 10 ms / 32 | 9.14 | 99.74% | 86.25% | 0.28% |

Increasing gain modestly improves the faster setting's edge returns, with some
increase in residual ripple. **12 ms / gain 32 is not a qualified operating
default.** The separate 20 ms / gain 16 clean-reference three-repeat comparison
below subsequently failed its strict preservation gates.

The 19 ms refinement had anomalously poor edge returns (~83.55%) despite its
longer exposure. The response is not strictly monotonic; interpolation alone
cannot qualify a setting. Prior 7–10 ms tests in another view lost 17–24% of
baseline-stable pixels. Those results remain valid and demonstrate why a
single favorable orientation cannot establish general preservation.

Stationary-block temperature peak: CPU 61.1°C / NPU 60.8°C. These are chip
sensors, not projector/optics temperatures. See P5 for the later raw soak.

### Clean-reference repeats: candidate remains unqualified

`p1-clean-reference-repeats/` contains three 12 ms / gain 32 repeats interleaved
with four 20 ms / gain 16 references, with original 20 ms / gain 24 outside
both ends. `p1-clean-reference-analysis.json` explicitly uses stages 1 and 7
as its clean reference; the original-baseline report/maps remain separate.

- 12 ms candidate rates 7.742, 7.745, 7.751 fps; 40 unique frames each.
- Clean-reference baseline p90 drift 0.040 mm.
- Candidate p90 depth shift **0.1275 mm**, over the 0.10 mm gate.
- Edge returns **87.01–87.29%**, versus clean-reference 93.78–94.13%:
  below the 95% relative-retention gate.
- Temporal MAD median/p90 **0.025/0.060 mm**, versus reference 0.015/0.030:
  also over the 1.5× gate.
- Interior returns remain 99.89%; only 1–4 of 50,819 stable pixels are
  persistently absent. This does not cancel edge/noise failures.

These repeats are internally consistent but less favorable than the earlier
gain-isolation block. Do not select the favorable block and ignore them.
The subsequent bounded gain-order probe (`p1-gain-order.json`) freshly read
gain after frame/exposure changes: it stayed 16 or 32 as commanded and restored
to 24. No automatic gain reset was observed in that closed-stream test.
Future acquisition blocks additionally verify gain after timing writes and
after the settled sample set, to detect any in-stream drift.

An exploratory *single*, unbracketed clean reference in the earlier gain block
put 15 ms / gain 24 at 97.97% relative edge return, p90 shift 0.045 mm and
MAD 0.015/0.035 mm. That makes approximately **6 fps** the next conservative
candidate to qualify, **not an already established maximum useful speed**.
The next orientation was subsequently confirmed by the operator (see below).

### Second stationary view, 15:56–15:58 UTC

The operator rotated the object after being asked to rotate and stop it;
scanner placement was unchanged. `p1-next-view-15ms/` implements the planned
three-repeat 15 ms / gain 24 block, with four clean 20 ms / gain 16 references
and original 20 ms / gain 24 at both outer ends. All nine settled sets contain
40 unique complete frames. The full stream delivered 478 complete frames,
zero shorts and zero sequence gaps. Every stage verified gain after timing
writes and again after acquisition. Original settings were restored.

Against clean reference stages 1/7 (`p1-next-view-15ms-clean-analysis.json`):

- Candidate fps **6.1399 / 6.1371 / 6.1399**.
- Relative overall/interior returns ~99.70% / 99.98%; relative edge returns
  **96.638% / 96.576% / 96.642%**, above the 95% gate.
- Only **1 / 5 / 5** of 49,749 baseline-stable pixels persistently missing.
- Reference p90 drift **0.0475 nominal mm**; candidate p90 shift **0.1000 mm**,
  at the 0.10 mm gate, with essentially no margin.
- Candidate temporal MAD median/p90 **0.025/0.050 mm**, versus bracketing
  mean **0.01625/0.035 mm**: **1.538×/1.429×**. The median narrowly fails
  the unchanged 1.5× limit. Using the later 0.015 mm reference alone is worse,
  not an alternative route to passing. Three consistent repeats do not erase
  this failure.
- CPU/NPU peaks **59.4/59.1°C**.

The original-reference analysis is separately retained. Its candidate p90
shift is 0.320–0.325 mm, over its 0.15 mm drift-adjusted gate. This is another
original-baseline bias failure, not a discarded comparison.

Fixed patches in `p1-next-view-plane-rois.json`, selected from reference depth
maps before fitting, reproduce the original setting's shape artifact:

| Median-map plane RMS | Original 20 ms / 24 | Clean 20 ms / 16 | 15 ms / 24 |
| --- | ---: | ---: | ---: |
| Triangular side | 0.185–0.189 mm | 0.0345–0.0353 mm | 0.0630–0.0631 mm |
| Block front | 0.3080–0.3083 mm | 0.0297–0.0305 mm | 0.0795–0.0806 mm |
| Base front | 0.2128–0.2132 mm | 0.0332–0.0337 mm | 0.0449–0.0461 mm |

In the fixed block-front rectangle, the original gain leaves **377/380** of
5,412 clean-reference-stable pixels persistently absent at the outer ends;
gain 16 references and all three 15 ms candidates have none missing there.
This is a visibly striped internal face defect, not merely a changing outer
silhouette. These are relative shape/return measurements, not calibrated print
accuracy. The top of the rotated target touches the sensor image boundary;
complete-part qualification requires reframing later, without moving mid-block.

`p1-next-view-refinement/` then tested 14 and 13 ms / gain 24, each separated
by clean references, with original settings at both ends. All 372 deliveries
were complete, with no sequence gaps; peak CPU/NPU 60.2/59.5°C; all settings
and restoration verified. Clean-reference stages 1/5 had p90 drift 0.035 mm.

| Setting | Short-stage fps | Relative edge returns | p90 depth shift | Temporal MAD median/p90 | Noise ratio median/p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14 ms / gain 24 | 6.705 | 95.890% | 0.055 mm | 0.0225/0.050 mm | 1.286×/1.333× |
| 13 ms / gain 24 | 7.029 | 94.323% | 0.040 mm | 0.0250/0.0525 mm | 1.429×/1.400× |

Neither setting lost significant stable interior depth (0 versus 6 pixels).
14 ms meets these numerical clean-reference gates in this **single** block;
13 ms fails edge retention despite better flatness. Original-reference bias
failures are retained (~0.32 mm p90 at both settings); no faster default has
been adopted. Only 11 of the 50,615 clean-reference mask pixels lie below row
276, where the visible base ends. At 13 ms, edge returns on rows 240–275 fall
to 75.79% versus 87.96% in the reference; the failure cannot be dismissed as
the large disconnected background/turntable region in the full loss plot.

#### Three-repeat 14 ms result

`p1-next-view-14ms-repeats/` interleaves three 14 ms / gain 24 candidates with
four clean references, retaining original settings outside both ends. All
478 deliveries were complete, with zero short frames or sequence gaps.
Nine settled sets contain 40 unique frames each; all gain/timing checks and
restoration succeeded. Peaks CPU/NPU **60.7/60.4°C**.

Clean-reference stages 1/7: 50,616 mask pixels, 49,758 stable; p90 reference
drift 0.045 mm. The fixed numerical gates are met in all three comparisons:

| Repeat | Relative overall / interior / edge returns | Stable pixels persistently lost | p90 depth shift | MAD median/p90 ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 | 99.576% / 99.971% / 95.310% | 0 | 0.0575 mm | 1.500× / 1.357× |
| 2 | 99.590% / 99.976% / 95.422% | 0 | 0.0600 mm | 1.500× / 1.357× |
| 3 | 99.568% / 99.974% / 95.187% | 1 | 0.0550 mm | 1.500× / 1.429× |

Candidate median MAD is 0.0225 mm versus reference 0.015 mm: **exactly at the
allowed 1.5× limit**, not below it. Edge retention has only 0.19–0.42 percentage
points of margin over the 95% gate. This is a **narrow, stationary, current-view
numerical pass relative to the clean reference**, not robust multi-view,
complete-part, absolute accuracy, sustained GUI or handheld qualification.

The 39-gap endpoint rates are 6.7085/6.7063/6.7089 fps. To expose their paired-
cadence phase bias, the two overlapping 38-gap same-phase endpoint estimates
are 6.5956–6.6001 fps. Therefore describe the candidate as **approximately
6.6 fps**, not a precisely established sustained 6.71 fps. All 40 samples are
retained; this diagnostic does not drop acquisition data or thin the pipeline.

Repeated median-map plane RMS at 14 ms: triangular side 0.0498–0.0513 mm,
block front 0.0437–0.0448 mm, base front 0.0363–0.0371 mm. Original settings
again produce much stronger rippling. Nevertheless, the preserved original-
reference comparisons **fail both edge retention (94.79–94.94%) and bias
(0.330–0.3325 mm p90)**. The clean-reference pass does not erase those failures.

The 16:06 UTC final inventory in `firmware/post-second-view-inventory/` freshly
verifies original 20,000/20,000 µs, auto-exposure 0, gain 24, unchanged firmware
version and depth-intrinsics hashes. Final CPU/NPU reads **59.8/59.5°C**.
This session captured **1,328 complete deliveries**, with **1,000 settled raw
frames** preserved across its three blocks; no shorts or sequence gaps. The
pre-capture diagnostic regression passed **51 tests**. No production code,
firmware, projector power, display settings or live mesh behavior changed in
this second-view session. No acquisition remains running.

`source-snapshot-second-view/` and `evidence-manifest-second-view.json` preserve
this phase's code/instructions and file hashes without replacing the earlier
snapshot or manifest.

## P2: randomized host ablation and implemented optimization

`p2-host-ablation/report.json`: CPU/GPU normals × synchronous/asynchronous
coverage × packed depth preview on/off, three randomized repetitions, 179
identical recorded frames each. All **4,296** decisions match the recorded CPU
reference exactly. Matching coverage snapshots are byte/value identical.
This proves computational equivalence, **not geometric correctness** (see P3).

With packed preview enabled, median offline frame-loop throughput:

| Normals | Coverage computation | Offline fps |
| --- | --- | ---: |
| CPU | synchronous | 6.24 |
| CPU | asynchronous | 7.13 |
| GPU | synchronous | 7.65 |
| GPU | asynchronous | 8.78 |

These include preparation/tracking/coverage/packed-preview computation, but
exclude raw acquisition, disk writes, RGB decoding and Qt widget presentation.
They are not live GUI fps. Nested stage timings are inclusive, not additive;
main-process CPU time excludes the coverage subprocess. Host load is recorded.

GPU-path profiling showed about 1.79 s of garbage collection and 6.0 s inside
normal construction per 179-frame production-mode replay. The normal path
converted an already packed covariance array into nested Python lists, then
immediately back into the same float64 array for upload.

Implemented the minimal removal of that round trip in `gpu_normals.py`.
No shader math, filtering, tracking gate, sample count or GPU fallback changed.
Real-GPU tests confirm exact list-versus-array eigensolves and full normal-field
agreement with the CPU. An offline test specifically rejects reintroduction of
the Python-list handoff.

`p2-packed-covariance/`: post-change production-mode replays took
18.087 / 18.327 / 18.677 s, versus 20.126 / 20.398 / 20.387 s before.
Median throughput improved **8.78→9.77 fps (~11.2%)** in this scoped replay.
Median GC time fell 1.79→1.05 s. All 537 post-change decisions are exact;
all 132 comparable cross-version coverage snapshots are identical. No live
performance claim is made from this offline improvement alone.

### Live GUI verification and display-power confound

Both new 8 ms live runs requested 300 deliveries, retained/accepted all 299
complete frames, rejected one short delivery, and exited normally with
verified original timing restoration. No experimental live mesh was enabled.
The turntable was last confirmed stopped; no new orientation or controlled
motion was assumed. These are short throughput checks, not P3 motion tests.

| Desktop state | Delivery fps | Sequence gaps | Depth previews applied | Model snapshots applied |
| --- | ---: | ---: | ---: | ---: |
| Monitor powered off | 11.08 | 0 | 36 / 299 | 30 / 75 |
| Monitor briefly awakened | 10.42 | 8 | 299 / 299 | 75 / 75 |

Reports: `p2-packed-live-8ms.json`, `p2-packed-live-display-on-8ms.json`;
child reports/captures `20260907-031140` and `20260907-031557` (local time).
The second run's saved-frame timestamps span 28.688 s: **10.388 complete fps**.
Its depth-preview channel applied every complete frame; the accumulated model
updated ~2.6 times/s. The 10 fps target is demonstrated for this short depth
preview run, **not a 10 Hz accumulated model or sustained moving qualification**.

The first run's ~1.3 Hz display updates cannot be reported as 11 fps visible
feedback. `xset q` confirmed the monitor was off. Briefly waking it restored
full preview delivery, supporting desktop presentation throttling as the cause
in this comparison. The original monitor-off state was restored in `finally`;
no display power-management configuration or graphics driver was changed.

Awake-run instrumentation: median tracking **70.09 ms**, preparation **20.18 ms**,
depth display/health slot **31.21 ms**, model-application slot **9.00 ms**,
OpenGL `paintGL` **0.605 ms**. These run on different threads and/or overlap;
they must not be summed as one serial frame time. The 20 ms GUI timer's gap
median/p95/p99/max was 21.46/60.89/81.31/94.99 ms. Both renderer and normal
compute identified the RX 9070 XT; 594 normal dispatches completed in 1.524 s.
Peak scanner CPU/NPU was 62.0/61.7°C. Existing RGB malformed-grab handling
remains visible: 156 pictures applied, 157 grabs without a usable picture.

There was no same-scene pre-optimization live A/B, so the controlled **11.2%**
speedup remains an offline result; do not attribute the full difference from
earlier moving GUI runs to this single change.

## P3: accepted poses can still drift badly

Source: prior real five-minute 8 ms GUI capture
`/home/ben/scans/scan/20260907-001402`, 2703 complete frames. No new physical
motion was invented, and no raw frame was changed or reconstructed into a mesh.

`p3-existing-motion-replay/`: first 1000 source frames, temporal strides
1/2/3/4, original 900-point preparation and unchanged live GPU tracker. Every
offered frame was accepted: 1000/1000, 500/500, 334/334, 250/250. Nonetheless,
individually image-matched revisits disagree under their recovered poses:

| Stride | Weakest individually matched agreement within 1.5 nominal mm |
| --- | ---: |
| 1 | 25.6% |
| 2 | 12.8% |
| 3 | 10.9% |
| 4 | 9.0% |

The original constant-lag coherence check incorrectly passed the stride-1
subset at 79.2%. Its period estimator chooses best matches at different
multiples of a revolution, takes their median lag, then tests different pairs.
The new **diagnostic** audit uses each image matcher's actual best pair, verifies
the source hashes, and preserves the legacy result alongside it. No production
tracking gate was silently changed. Synthetic tests prove it reads the chosen
pair, detects a known depth shift and rejects changed input hashes.

Concrete stride-1 example: frames **166 and 918** have raw median image
difference **0.120 mm**, but the recovered poses project with median depth
offset **8.01 mm**, only **25.6%** within 1.5 mm. Their relative camera
translation includes **−10.29 mm in z**. Directly aligning the two prepared
frames independently still finds a 94.4%-matched fit at 0.239 mm point-to-plane
RMS. The acquisition contains matching geometry that the accumulated chain
misplaces; accepting references which share the same drift is not validation.

`p3-full-five-minute-replay/` and `p3-full-revisit-audit.json` extend this to
**all 2703 frames**: every frame accepted, weakest actual matched-pair
agreement only **11.9%**. A CAD accuracy result on this misplaced aggregate
would be misleading and is withheld.

### Held-out fixed-axis experiment

`p3-turntable-heldout.json` fits the existing turntable-specific solver only
to 200 frames sampled from the first 400 source frames (~1.5 rotations), using
recorded delivery times, then predicts all later poses without refitting them.
Three fitting rounds reduce axis scatter to 0.10° and final centre change to
0.16 mm. Training point-to-plane medians: 0.07 mm adjacent, 0.28 mm at ~60°,
0.47 mm at ~90°, 0.13 mm around a turn.

On the same individually image-matched full-capture pairs, weakest held-out
agreement improves **11.9%→67.5%**, but still misses the 75% gate. This is
useful evidence for a turntable constraint, **not a passed reconstruction or
a handheld fix**. It does not establish constant physical speed or solve
general six-degree-of-freedom tracking.

Priority after this phase: qualify exposure/gain over additional views, and
test independent revisit/anchor constraints and drift detection before calling
a higher accepted-frame rate a higher useful scanning speed.

## P5: 15-minute maximum-tested raw acquisition endurance

`p5-15min-7ms.json` extends the earlier successful five-minute 7 ms soak.
This is explicitly an **acquisition-only** thermal check, not a quality-qualified
scan. No experimental mesh or GUI was running. Temperatures were freshly read
every five seconds, with the same 80°C stop; protections remained enabled.

- Elapsed 900.024 s; **11,650/11,650 complete, unique depth payloads**.
- **12.9500 fps**, zero short frames and zero sequence-reported missing frames.
- Each five-minute block remained at approximately 12.95 fps; no observed
  thermal throughput reduction during this run.
- Delivery-gap median/p95/p99/max: **93.59 / 100.47 / 101.97 / 104.60 ms**.
  Mean is 77.22 ms because the stream alternates shorter and longer gaps.
- CPU 58.0→**67.8°C**, NPU 57.7→**67.0°C**; final values were the peaks.
- Last-five-minute least-squares temperature slopes: **+0.204°C/min CPU**,
  **+0.215°C/min NPU**, from 59 samples. This is not an indefinite plateau.
- 150 raw excerpts, ten each minute, preserved with individual SHA256 hashes.
  Their presence does not qualify thermal optical accuracy or scene stability.
- Fresh readback verified 7/7 ms before restoration and original **20/20 ms,
  auto-exposure 0** afterward. Process exited normally.

The earlier depth-retention failures at 7 ms remain failures. These measurements
show that this USB2 connection can sustain ~13 raw depth fps and that chip
temperature was not the rate limiter in this run; they do not establish a
quality-preserving maximum or a safe projector temperature. A 30-minute raw
run was not automatically escalated. Candidate endurance and the deliberate
five-cycle lifecycle test remain separate pending stages.

## P6: firmware evidence and flash gate

Official pinned SDK downloaded for **offline analysis only**, never loaded as
an executable scanner driver: 94,574,928 bytes, SHA256
`49e0865c7f9e130ddfde5948e658190143ffb8b21e9583877d14e3e17cdae3c4`,
matching the repository manifest. Artifact `firmware/lib3DCamera-pinned.so`.
[Pinned official source](https://github.com/Revopoint/3DViewer/blob/516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7/thirdparty/3DCamera/linux/x64/lib3DCamera.so).

Fresh bounded inventory at 07:31 UTC, `firmware/inventory/`:

- `/data/firmware_version` and `/userdata/firmware_version` both contain
  `v2.8.7.20220610`. Gain getter `0x9a3` freshly returns **24**.
- `/data/camparam/Pl.bin` is 40 bytes, SHA256
  `fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf`.
  This preserves depth intrinsics, **not all device calibration**.
- SPI NAND MTD partitions: vnvm 1 MiB, uboot 4 MiB, misc 1 MiB, boot 8 MiB,
  rootfs 80 MiB, recovery 16 MiB, userdata 6 MiB, oem 8 MiB, backup 896 KiB.
  Root is read-only UBIFS; `/userdata` and `/oem` are mounted read/write;
  `/tmp` is RAM. Names alone do not prove a working rollback mechanism.
- Known model/config paths returned zero bytes. This read protocol cannot
  distinguish an absent file from an empty file; neither is invented as data.

Exact controller disassembly binds sensor gain to setter `0x903` at ARM
`0x21c18`, getter `0x9a3` at `0x21c70`. The depth gain handler `0x19844`
checks UI gain 1..16, shifts left four bits, clips to 255 and calls that
setter. Only raw register 16/24/32 was tested, each with fresh named readback.
All changes were volatile and restored; no projector-current manipulation.

SDK `setFringePattern` at `0x1fd5fa` rejects old/pre-UVC models or firmware
below 2.7.35; enum patterns **5/6** additionally require firmware ≥2.9.43.
Pattern **6 only** selects 6600 µs below 2.9.45 and 6200 µs thereafter.
This is not evidence that 2.9.43/45 is available for this MINI, nor permission
to try unsupported pattern IDs. Current 2.8.7 fails that SDK version gate.

September 9 static follow-up: this conclusion is specific to the pinned Linux
binary. The newer Windows library's corresponding version predicates apply
only to POP 2, not this serial's MINI model ID. The
[package cross-check](firmware-package-review-20260909.md) preserves both exact
listings; host permission is still not device implementation or scan quality.

The SDK update path contains upload of `/tmp/webroot/Firmware.img` and an
updater executable, then a flash command. No package or updater was uploaded
or run. A recovered controller executable is not a full firmware backup.

The [official download matrix](https://www.revopoint3d.com/pages/support-download)
lists original MINI under desktop Revo Scan **5**, not desktop Revo Scan 6;
newer MINI 2 support must not be mistaken for MINI support. The
[manufacturer's support reply](https://forum.revopoint3d.com/t/mini-firmware-update-how/21792/3)
requests exact model and installed version for update assistance. No exact,
authenticated replacement image with a demonstrated recovery route has yet
been obtained. Official PDF endpoints returned HTTP 503 during this session;
forum button-reset anecdotes are not treated as verified recovery instructions.

**Flash branch currently gated:** need exact-compatible image and release
benefit, complete device-specific calibration preservation, a supported local
updater/recovery path and physical recovery access. No vendor message was sent.
Runtime experiments and host tests continue independently of that gate.

## First-phase verification and remaining execution

The first phase's final post-GUI inventory at **08:19 UTC**, `firmware/post-gui-inventory/`,
freshly verifies original **20,000/20,000 µs, auto-exposure 0, gain register 24**.
Firmware-version and depth-intrinsics hashes match the initial inventory.
Final CPU/NPU reads were **59.8/59.5°C**. No scanner acquisition or benchmark
process is left running. The monitor's initial powered-off state was restored.

Validation after the production array-handoff change: **384 tests passed**
across live acceleration (including real-GPU parity), async coverage, stationary
and firmware probes, revisit diagnostics, preparation, tracking and GUI
Stop/feedback/surface behavior. Two subsequently added evidence-manifest tests
also passed with the three revisit/plane tests (five tests total). Ruff passed
on selected implementation/diagnostic/test files; mypy passed on 12 selected
source/diagnostic files. This is a targeted run, not a claimed full-repository run.

Temporary live reports/screenshots, harnesses, prior reference report and raw
five-minute soak report are copied into the diagnostic root. A source snapshot
and `evidence-manifest.json` preserve code and SHA256/size records for this
phase's diagnostic files and both new live raw captures. Hashes establish file
integrity, not physical accuracy. Existing moving-replay input hashes remain
in their respective reports; original captures are not moved or overwritten.

The previously proposed second-view 15 ms block is now completed, as recorded
above. Next operator-dependent block, **not yet executed**: rotate the target
another approximately quarter turn and stop it. Predeclare three **14 ms /
gain 24** candidates bracketed by clean 20 ms /
gain 16 references, retaining original 20 ms / gain 24 at the outer ends:

```bash
.venv/bin/python tools/scanner_stationary_quality.py --execute \
  --output /home/ben/scans/diagnostics/20260907-thorough/p1-third-view-14ms \
  --timings 20000 20000 14000 20000 14000 20000 14000 20000 20000 \
  --gains 24 16 24 16 24 16 24 16 24
```

Analyze both the original outer baseline and the separately declared clean
reference stages 1/7, using the unchanged gates. Inspect framing first: the
second view's top touched the image boundary. This is still only the next
view; distance, lighting and confirmed part identity
remain required. Follow with fresh repeated low/medium/max turntable captures
and independently timed handheld passes, not just recorded-data strides.
Quality-qualified 5/15/30-minute endurance, warm/cool quality checks, bounded
start/stop reliability, independent drift constraints and the firmware flash
gate are explicitly unfinished. No quality-preserving maximum handheld speed
has been established.

## P1: front-side revisit and arm-warning replacement, 16:13–16:27 UTC

The operator rotated again but was unsure of the direction. Visual comparison
in `p1-orientation-check-03/` appears to show the original front side again,
slightly repositioned, not a third distinct view. An exploratory two-frame
registration also supports a revisit: ~0.39° rotation, ~9.07 nominal mm
translation, 83.95% matched and 0.264 mm point-to-plane RMS. This is not the
physical rotation angle, a quality gate or absolute pose ground truth.

The operator then reported that an arm entered a scan. Exact affected frames
are unknown. Both that view probe and `p1-front-revisit-14ms/` have preserved
`quality-validity.json` annotations **withholding qualification**. The latter
had already completed before the stop request reached its process: 478
complete deliveries, 360 settled raw frames, zero short/sequence-missing
frames, restored settings. Completion does not establish an uncontaminated
scene. Neither original capture nor report was rewritten or deleted.

`p1-orientation-check-04/` is a fresh, read-only 32-frame baseline check at
16:21:55–16:22:05 UTC: all 32 complete and unique, no short frames or sequence
gaps, maximum CPU/NPU 58.9/58.6°C, original timing/gain freshly verified.
`p1-arm-warning-audit/` compares individual frames, retaining full-frame plots,
per-frame hashes and added/missing/shifted-depth counts. No large passing
surface was identified in the retained frames inspected; this does **not**
disprove the operator report, recover unsaved startup frames or exclude a
stationary contaminant. The flagged batch remains excluded regardless.

### Replacement 14 ms block

After checking the fresh reference for a stationary, unobstructed scene,
`p1-front-revisit-14ms-replacement/` repeated the exact nine-stage sequence:
original 20 ms / gain 24 at stages 0/8, clean 20 ms / gain 16 at stages 1/3/5/7,
and 14 ms / gain 24 at stages 2/4/6. All 478 deliveries were complete; zero
short frames or sequence gaps. All 360 settled payloads were unique. Every
gain/timing transition and ending gain matched its requested value. Maximum
CPU/NPU temperatures were **60.7/60.0°C**; original settings restored.

`p1-replacement-scene-audit/` preserves per-frame inspection evidence. The
largest additional connected surface outside the fresh reference's five-pixel
margin was 20 pixels, with no large new occluder in the inspected plots. These
are scene-change diagnostics, not automatic semantic arm detection. The
reference component has only two pixels of top margin; no border pixel is
occupied in that baseline. Full-frame data and support/background returns are
retained, not cropped away to obtain a pass.

Clean-reference analysis uses the same predeclared stages 1/7 and unchanged
gates: mask 50,328 pixels, 49,164 stable pixels, baseline-to-baseline p90 shift
0.050 mm. Relative coverage uses the mean of the two reference measurements.

| 14 ms repeat | Overall retention | Interior retention | Edge retention | Persistent stable loss | p90 depth shift | Temporal MAD median / p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 99.488% | 99.932% | 95.449% | 0 pixels | 0.0600 mm | 0.0225 / 0.0500 mm |
| 2 | 99.505% | 99.932% | 95.622% | 0 pixels | 0.0600 mm | 0.0225 / 0.0500 mm |
| 3 | 99.474% | 99.931% | 95.310% | 0 pixels | 0.0575 mm | 0.0225 / 0.0500 mm |

All three meet the **numerical clean-reference gates narrowly**. Median MAD
is exactly 1.5× the reference; p90 MAD is 1.429×. Edge margin is only
0.31–0.62 percentage points above the 95% requirement. Rates over 39 gaps
are 6.707/6.705/6.703 fps, but phase-balanced overlapping 38-gap measurements
are **6.596–6.601 fps**: describe this as approximately **6.6 fps**, not a
sustained 6.71 fps result. No tracking, GUI, RGB or live meshing ran here.

The separate original-reference comparison, preserved in the spatial report,
still **fails**: candidate p90 shifts 0.370–0.375 mm versus a 0.190 mm gate.
The favorable clean-reference outcome does not erase the original baseline's
different shape/holes or establish absolute print accuracy.

Full-resolution maps are in `p1-front-revisit-14ms-replacement-spatial/`;
the separately labeled clean analysis is
`p1-front-revisit-14ms-replacement-clean-analysis.json`. Fixed patches were
chosen from the new baseline before candidate plane fitting, recorded in
`p1-front-revisit-plane-rois.json`. Plane reports retain all four patches:
ramp RMS improves from ~0.163 mm original to ~0.028 mm clean and
0.030 mm candidate; right face from ~0.168 to 0.026 and 0.043 mm. The selected
left patch remains poor (~0.51–0.52 mm clean/candidate, ~0.59 mm original)
and the base ~0.08 mm clean/candidate; neither is silently discarded or
presented as calibrated flatness. Independent fits only measure local shape;
they do not register away matched-pixel depth shifts.

### Restoration, tooling and remaining dependency

Fresh read-only `firmware/post-front-revisit-inventory/`, ending 16:27 UTC,
verifies **20,000/20,000 µs, auto-exposure 0, gain register 24**. Firmware
version and depth-intrinsics hashes match the prior inventory. Final chip
readings were **60.2/59.5°C**. No scanner acquisition is left running.

New view-check and offline scene-audit helpers preserve framing and transient
evidence. The quality analyzer now refuses a validity-annotated batch unless
its annotation explicitly allows qualification. **58 targeted tests passed**,
including synthetic additions/occlusion/depth changes and fail-closed operator
warnings; selected Ruff checks and mypy checks passed. Production tracking,
GPU arithmetic, display code, firmware and persistent scanner settings were
not changed in this continuation.

`source-snapshot-front-revisit/` preserves the selected capture/analysis code,
tests and updated documents. `evidence-manifest-front-revisit.json` records
SHA256/size for the diagnostic root, including earlier manifests without
rewriting them. Integrity hashes do not certify the physical scene.

This adds a repeated front-view result to the earlier side-view result, not
a third distinct orientation or a maximum handheld speed. Next physical step:
rotate approximately half a turn from the present front-facing position and
stop, then verify that the back/other side is actually visible and has adequate
framing before another identically bracketed 14 ms block. Motion, candidate
endurance and broader distance/lighting qualification remain pending.

## P1: third distinct view, 16:40–16:44 UTC

After the operator's next rotation, `p1-orientation-check-05/` captured 32/32
complete unique baseline frames, zero short frames or sequence gaps, with
freshly verified original timing/gain. CPU/NPU peaks were 58.5/57.7°C. Visual
inspection shows a distinct back/other-side view, not an exact measured angle.
The largest connected component spans `[116, 2, 466, 334)` and includes
support/turntable returns; the main target is not clipped but has only two
pixels of top margin. `p1-orientation-check-05-scene/` preserves per-frame
checks and plots, with no large transient occluder identified.

`p1-third-view-rois.json` records the fixed object-area rectangle
`[110, 0, 475, 278)` and four planar patches **before** the quality block.
The crop is not semantic segmentation. Full connected-region outcomes and
uncropped maps are preserved alongside cropped metrics.

`p1-third-view-14ms/` repeats the same nine-stage sequence: original 20 ms /
gain 24 at the outer ends, clean 20 ms / gain 16 at stages 1/3/5/7, and three
14 ms / gain 24 candidates at stages 2/4/6. All **478 deliveries complete**,
zero short/sequence-missing frames, **360/360 globally unique settled payloads**.
Requested timing/gain and ending gain verified at every stage; original
settings restored. Maximum CPU/NPU **60.2/60.0°C**.

Clean-reference stages 1/7 define 51,691 connected pixels, 50,674 stable;
the predeclared crop retains 51,478 pixels, 50,633 stable. Baseline p90 drift
is 0.045 mm in both. Each of the following meets the unchanged numerical gates:

| 14 ms repeat | Full-region edge retention | Cropped edge retention | Persistent stable loss | p90 depth shift | Temporal MAD median / p90, crop |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 96.749% | 96.430% | 1 pixel | 0.0450 mm | 0.0250 / 0.0500 mm |
| 2 | 96.670% | 96.418% | 0 pixels | 0.0475 mm | 0.0250 / 0.0500 mm |
| 3 | 96.670% | 96.385% | 3 pixels | 0.0450 mm | 0.0250 / 0.0500 mm |

Cropped overall retention is 99.616–99.622%, interior 99.984–99.990%.
MAD is 1.429× reference median and 1.333× reference p90 in the crop (the first
full-region repeat's p90 is 1.4×). Phase-balanced rates are **6.597–6.599 fps**.
These are stationary comparisons, not tracking or GUI throughput results.

The **original-reference comparisons still fail**. Full connected-region
relative edge coverage is ~75.3–75.4%, with ~8.0% stable loss; much of that
region lies on the support. The predeclared object-area crop reduces stable
loss to 131–134 pixels (~0.26–0.27%), but edge retention remains **93.73–93.76%**
and p90 shift ~0.210 mm exceeds its 0.145 mm gate. Reporting the crop does not
erase either failure or make the original baseline a metrology oracle.

Artifacts: `p1-third-view-14ms-{clean,clean-roi,original-roi}-analysis.json`,
`p1-third-view-14ms-spatial/` (also the original full-region report), and
`p1-third-view-14ms-scene/`. The scene audit hashes every retained frame;
largest added component beyond the fresh baseline's five-pixel margin is
17 pixels. No large new occluder appears in inspected frames; this is not an
automatic semantic guarantee.

`p1-third-view-14ms-planes.json` retains all four preselected patches. Ramp
RMS: ~0.038–0.040 mm original, ~0.0165 mm clean, ~0.017 mm candidate. Left
face: ~0.31–0.36 mm original, ~0.042 mm clean, ~0.064 mm candidate. Left top:
~0.009–0.010 mm in all settings. The **base patch fails the planarity criterion**
and remains explicitly invalid; no replacement patch was selected afterward.
The report helper now retains such failures and continues other patches rather
than aborting the entire report. Fits do not correct candidate depth bias.

Three observed orientations now support 14 ms / gain 24 as a **provisional
stationary finalist around 6.6 fps**, with tight margins in the earlier views.
This does not establish a universal depth-preserving maximum, complete-part
accuracy or reliable handheld tracking. The planned five-minute candidate
shakedown and immediate warm repeat follow separately below.

## P5: 14 ms five-minute acquisition and post-stream USB failure, 16:49–16:54 UTC

`p5-5min-14ms.json` records the first five-minute shakedown of the provisional
stationary finalist. The closed timing selector was extended to allow only
7 or 14 ms (default remains 7); gain 24 is freshly checked before and after
acquisition. No gain, power, clock, firmware or production algorithm change.

- Acquisition elapsed **300.0245 seconds**; **1,975 complete, unique payloads**.
- **6.59824 fps**, zero short frames or sequence gaps. No observed rate decline.
- Full recorded-gap mean 151.556 ms, conventional median 150.962 ms,
  p95/p99/max **248.450 / 249.665 / 251.688 ms**. The existing stream summary
  reports the upper-middle sample, 240.889 ms, rather than averaging the two
  middle gaps; the difference is large because cadence alternates short/long.
  Both original summary and individual records are preserved; no production
  statistic was changed here.
- Initial chip readings 58.5/57.7°C; last and peak CPU/NPU **62.5/62.2°C**.
  Last-minute fitted slopes ~+0.63/+0.83°C per minute are quantized and still
  rising, not an indefinite plateau. Temperatures are not projector measurements.
- Fifty raw excerpts, ten at each minute start, retained with SHA256 hashes.
  All fifty hashes and full payload lengths verified offline.

The **complete experiment failed during final verification**. After streaming
closed, fresh getters confirmed 14/14 ms and gain 24. Restoration then wrote
20/20 ms and a fresh timing read confirmed **20/20 ms, auto-exposure 0**.
The following final gain read failed while clearing the bounded file-read
selector: UVC unit 4/control 1 returned EPROTO, then ENODEV. Thus final gain
verification and final post-restoration temperature reads did not complete;
`restoration_verified` is **false**, not an assumed success.

At **16:54:23 UTC**, the kernel records this control failure and USB 1-1
disconnect (device 83). Automatic host re-enumeration attempts 84–87 then fail
with descriptor timeout/protocol errors and address setup failures. Exact
filtered logs are in `p5-5min-14ms-usb-kernel.txt`. The device remained absent
in read-only USB/serial/video-node checks through 16:59 UTC. No agent-issued
USB reset, driver detach, power cycle, flash or blind retry loop was attempted.

This localizes the observed failure to post-stream control/lifecycle handling;
it does **not** prove its electrical/firmware root cause. Uninterrupted
acquisition at 6.6 fps worked and chip readings were far below the test cutoff,
but that alone cannot exclude another component or USB/control-path fault.

The original report's `status` says `completed` for acquisition while its
restoration flag is false. Preserve it unchanged. The separate
`p5-5min-14ms-outcome.json` explicitly marks the overall experiment failed.
The harness is corrected for future runs to set overall status failed on a
restoration/readback exception, preserving the preceding status separately.
A synthetic final-gain failure test exercises precisely this distinction.
The longer-stage predecessor gate rejects this report because restoration
did not verify. No 15/30-minute candidate escalation ran.

### Limited excerpt comparison, not the planned warm repeat

The immediate nine-stage warm block **was not executed** because the scanner
disappeared. A later cooled/reconnected capture must not be relabeled as that
immediate warm comparison.

As a limited offline check, compare each ten-frame minute-start excerpt to the
initial third-view stage 4's forty 14 ms frames, using the initial clean
references' fixed object-area mask. P90 absolute shifts for minutes 0–4 are
**0.0600, 0.0475, 0.0825, 0.0725, 0.0875 nominal mm**. Newly persistently lost
initial-stable pixels are 0/0/12/3/15 out of 50,428; relative edge returns range
99.556–100.101%. These smaller, unbracketed samples do not replace the
forty-frame three-repeat comparison, establish thermal causality or represent
the final minute's end. Minute 0 also includes startup. No data were aligned
or refitted to remove shifts.

New offline `stationary_thermal_compare.py` retains fixed initial masks and
signed shifts/losses; its full warm-block entry point is not claimed executed.
**70 targeted tests passed**, including requested 14 ms timing, restoration on
stream failure, final-gain failure status, retained invalid planar patches and
known warm depth shifts/occlusion. Ruff and mypy passed on the selected changed
diagnostics. Production GUI/tracking/GPU code was not changed in this turn.

`source-snapshot-third-view/` preserves the selected post-run diagnostics,
tests and updated documents, including the reporting correction made after
the failure. `evidence-manifest-third-view.json` adds SHA256/size records for
the diagnostic root without rewriting previous manifests or captures.

Recovery dependency: physically reconnect the scanner USB, leave the target
and turntable stationary, then perform fresh serial/state/temperature checks.
The current device state cannot be freshly verified while it is absent.
Retain this lifecycle failure and start a separately labeled sequence; do not
claim completion of the full test plan or a proven handheld speed.

## Post-replug lifecycle isolation and firmware evidence, 17:16–17:53 UTC

The operator replugged the scanner. Fresh read-only inventory in
`firmware/post-replug-inventory/` verified the exact serial, original
20/20 ms, auto-exposure 0, gain 24, and unchanged firmware/depth-intrinsics
hashes. CPU/NPU readings were 57.2/56.9 to 57.6/57.3°C. The prior failed
five-minute run and its unavailable immediate warm comparison remain unchanged.

The first new baseline exposed a second, distinguishable behavior: even with
no timing writes, the USB device briefly disappears after stream shutdown,
then returns automatically at a new address. The original five-case matrix
was stopped after that first case. Separately predeclared isolation branches
followed; these are not five consecutively stable lifecycle passes.

| Artifact suffix under `p5-lifecycle-` | Controlled test | Outcome |
| --- | --- | --- |
| `01-baseline` | 64 baseline frames; immediate post-stop getters | Readbacks succeeded, but USB 88→89 occurred after the original report closed; the added outcome sidecar marks topology unstable |
| `02-held-baseline` | Same test with a verified read-only usbfs descriptor held | USB 89→90 still occurred while held; runtime active/usage 1 and suspended-time counter unchanged immediately before disconnect |
| `03-inspect` | No stream, fresh state/gain/uptime | Original state and USB 90 stable; device uptime continued across the previous restart |
| `04-quiet-baseline` | No timing writes and no UVC after stop, usbfs held | USB 90→91 at +1.223 s after STREAMOFF; post-return verification deliberately separate |
| `05-post-quiet-inspect` | Controller log before verbose getters | Fresh original settings; runtime log captures bulk timeout, gadget shutdown and reopening |
| `06-live-quiet` + `07-post-live-inspect` | Restore/verify original timing while still streaming; quiet stop; separate fresh inspection | Both state checks passed; USB 91→92 remains a topology failure under the original no-change criterion |
| `08-settled-post` | 64 frames at 14 ms; six seconds quiet; fresh readback/restoration after stable return | Passed the separately declared recovery-boundary criterion, explicitly retaining USB 92→93 |

All five acquisition cases delivered 64 complete unique frames with zero short
frames or sequence gaps. The live-restoration case mixes 14 and 20 ms frames;
its fps is not a single-setting performance result. No returned UVC transport
error occurred in these short cases. A successful readback alone does not make
the earlier topology failures passes. Holding the read-only usbfs descriptor
did not prevent restart; no power policy, USB ioctl, interface claim, driver
binding or reset was changed. Device `MemAvailable` was roughly 678,000 kB in
the first baseline; there is no evidence here of an out-of-memory failure.

### What the running firmware actually did

`p5-firmware-stop-analysis/` preserves hash-bound offline ARM disassembly,
selected kernel/controller logs and the detailed evidence chain. The controller
is the same 166,412-byte ELF recovered in P0, not a replacement firmware image.

In the quiet baseline, host STREAMOFF returned at **17:45:23.553722 UTC**;
the kernel recorded disconnect at **17:45:24.776864**, 1.223142 seconds later.
The freshly captured controller log records `bulk select timeout`,
`bulk_timeout exit`, depth stop, exit of all gadget threads, then reopening
DepthCam and RGBCam. Continuous device uptime distinguishes this brief USB
restart from a whole-device reboot.

In the pinned controller, the worker prepares a one-second `select` timeout
at 0x15b3c–0x15b5c. Its timeout/exit sequence stops the DSP, waits 100 ms,
stops streaming, signals the gadget manager and exits. The manager's condition
wait leads to all-thread teardown and a loop back to gadget discovery. Runtime
logs corroborate that teardown/reopen behavior. The separate disassembled
branch labeled `reboot` is **not** claimed to have run. Literal-pool words
rendered as instructions are not treated as execution evidence.

This explains the routine delayed restart, **not the exact low-level cause
of the earlier unrecoverable descriptor/address failure**. Immediate post-stop
controls can overlap this teardown; avoiding that overlap is a tested host
workaround, not proof of the old failure's cause or a repaired firmware.

### Diagnostic workaround and five-minute retry, 17:58–18:03 UTC

After the short settled-post case passed, the thermal harness gained an
opt-in `--quiet-after-stop` path. It performs no UVC requests for six seconds
after stop, requires the exact serial/USB identity stable throughout the final
two seconds, then resolves a fresh descriptor. The old USB epoch is retained
as evidence, not treated as continuously connected. A missing/flapping identity
or first transport fault prohibits further UVC cleanup/retries. This variant
is limited to 300 seconds; production GUI/stream behavior is unchanged.

`p5-5min-14ms-quiet-retry.json`, after the separately captured fresh initial
quality block, records:

- **300.02384 seconds, 1,975 complete unique frames, 6.598294 fps**;
  zero short frames or sequence gaps, no returned transport fault.
- Gap mean 151.554 ms, conventional median 152.022 ms, p95/p99/max
  248.464/249.385/250.486 ms. The original stream summary's upper-middle
  median is 181.451 ms; both summary and individual intervals are retained.
- CPU/NPU start 59.4/58.6°C, peak/end **63.3/63.0°C**. Last-minute fitted
  slopes **+0.326/+0.179°C/min** remain positive, not a demonstrated plateau.
  All 59 thermal samples remain below the unchanged 80°C cutoff. They do
  not measure projector/optics temperature.
- All **50 raw excerpts** have verified full 512,000-byte length and SHA256.
- STREAMOFF at **18:03:00.481762 UTC**; cached sysfs first sees disappearance
  at +1.417 s and the new device at +1.869 s, USB 94→95. The trace contains
  **zero UVC requests during the first six seconds**; the first is +6.007843 s.
- Fresh returned-device state was 14/14 ms, gain 24. Original **20/20 ms,
  auto-exposure 0, gain 24** was restored and verified, followed by temperature
  readback and two seconds of stable final USB observation.

Thus this retry passes acquisition, the declared restart boundary and final
verification. It does not erase the first failed run or demonstrate repeated
long-run lifecycle reliability. Artifacts include the corresponding
`-summary.json`, `.uvc-trace.jsonl`, `-usb-kernel.txt` and `-raw/` directory.
No 15/30-minute candidate extension ran.

## P1/P5: fresh initial, warm and cooler quality, 17:55–18:15 UTC

`p5-retry-{initial,warm,cooldown}-quality/` each uses the unchanged nine-stage
sequence: original 20 ms/gain 24 at 0/8; clean 20 ms/gain 16 at 1/3/5/7;
14 ms/gain 24 at 2/4/6. Each stream delivered **478 complete frames**, no
shorts or sequence gaps, and retained **360 globally unique settled payloads**.
Each restored and verified original settings. No new object rotation was
requested between these blocks. The first new initial block, not the older
pre-replug capture, defines cross-block reference pixels.

The warm block's first thermal reading is about **24.9 seconds after soak
STREAMOFF**. Its candidate stages occur later in the approximately 95-second
bracketed block; this is not an instantaneous peak-temperature measurement.
The cooler block begins about 9.9 minutes after the warm block began, following
idle cooling. Its first CPU reading is 59.8°C versus initial 58.0°C, and it
warms during capture. Thermal history is not perfectly matched.

Unchanged numerical comparisons to each block's own clean references 1/7:

| Block | CPU/NPU start → peak, °C | 14 ms edge retention | 14 ms p90 shift | MAD ratio median / p90 | Clean numerical gates |
| --- | --- | ---: | ---: | ---: | --- |
| Initial | 58.0/57.7 → 60.7/60.0 | 96.593–96.908% | 0.0450–0.0475 mm | 1.333× / 1.313–1.375× | 3/3 pass |
| Warm | 62.5/62.2 → 63.8/63.0 | 96.737–97.058% | 0.0425–0.0450 mm | 1.429× / 1.290–1.355× | 3/3 pass |
| Cooler | 59.8/59.5 → 62.0/61.3 | 96.637–96.720% | 0.0450–0.0475 mm | 1.333× / 1.313× | 3/3 pass |

Phase-balanced candidate rates across these blocks are **6.595–6.601 fps**.
All 40 frames remain in quality statistics; fps uses the first 38 intervals
to balance the alternating delivery gaps. Overall relative returns exceed
99.65%, interior exceeds 99.98%, and persistent stable loss is 0–2 pixels
against each block's own clean baseline. Clean masks are 49,472 / 49,457 /
49,516 pixels; all fit inside the previously declared `[110, 0, 475, 278)`
rectangle, so full clean-region and cropped results are identical here.

The **original-reference failures persist**. In the fixed object-area crop,
candidate edge retention is 93.47–94.17% and p90 shifts 0.235–0.245 mm, beyond
each block's unchanged edge/bias gates. Full original-reference components
include more support returns: ~74–75% edge retention and ~8.5% persistent loss.
Full data, maps and both comparisons remain preserved; clean-reference passes
do not establish absolute accuracy or overrule the other failures.

### Cross-block shift and partial reversal

`p5-retry-warm-vs-initial.json` and `p5-retry-cooldown-vs-initial.json` use the
**same 49,472-pixel initial clean-reference mask**, never refitted to later
data. Each stage is compared to the matching initial timing/gain stage, with
no alignment or temperature correction. Values below are p90 absolute
matched-pixel median-depth shifts in **nominal mm**, not calibrated accuracy:

| Same stage/settings | Warm versus initial | Cooler versus initial |
| --- | ---: | ---: |
| First 14 ms / gain 24 repeat | 0.1650 | 0.0925 |
| Second 14 ms / gain 24 repeat | 0.1725 | 0.0950 |
| Third 14 ms / gain 24 repeat | 0.1600 | 0.0925 |
| Four clean 20 ms / gain 16 references, range | 0.1500–0.1650 | 0.0850–0.0950 |
| Original 20 ms / gain 24 endpoints | 0.2300–0.2600 | 0.1325–0.1525 |

Warm candidate signed medians are only +0.020 to +0.025 mm, with positive and
negative spatial changes: this is not simply a uniform translation. Candidate
new persistent losses are 27–36 initial-stable pixels when warm, falling to
0–2 in the cooler repeat. Fixed-mask candidate edge returns are 98.802–99.427%
of initial when warm, then 99.706–100.006% when cooler. Most coverage remains.

The depth shift **partly reverses as the scanner cools**, and affects the
slower clean reference as well as 14 ms. This supports a reversible warm-up-
associated component; one unmatched heat/cool sequence cannot distinguish
optical/calibration drift from small thermomechanical/fixture motion, nor
establish a correction model. Within-block numerical passes are not
temperature-invariant accuracy qualification. Retain the warm shift; do not
escalate the candidate to 15/30 minutes on this evidence alone.

### Scene, planar checks and durable evidence

The three scene audits hash all 1,080 retained frames. Inspected plots show
the same back/other-side part view, still with only a few pixels of top margin,
and no apparent large new occluder. Relative to the fresh clean stage-1
reference, largest added components in clean/candidate stages are at most
16 / 33 / 16 pixels in initial/warm/cooler blocks. Original gain-24 endpoints
instead add support components up to 4,865 / 4,877 / 4,868 pixels. They are
retained and visibly associated with the stronger support return; do not
mistake these diagnostics for semantic arm detection. Five-mm change counts
are small, but cannot rule out sub-mm fixture motion or a reference contaminant.

All four previously selected planar patches remain in
`p5-retry-{initial,warm,cooldown}-planes.json`. Clean/candidate ramp RMS is
~0.043–0.045 mm; left-face RMS ~0.041–0.044 / 0.059–0.061 mm, versus original
~0.33–0.37 mm; left-top ~0.017–0.020 mm. The broad base remains rough
(~0.31–0.34 mm clean/candidate). In these new data it clears the helper's loose
planarity rejection threshold, but is not a clean accuracy patch; the older
pre-replug invalid result is not relabeled. Fits do not remove cross-block shifts.

`p5-retry-quality-gates.json` preserves initial/warm numerical gates;
`p5-retry-cooldown-quality-gates.json` preserves cooler gates.
`p5-retry-comparison-summary.json` consolidates them with telemetry, scene
counts and fixed-initial-pixel comparisons. Raw reports remain the underlying
evidence; helper field `warm` denotes the second input even in the explicitly
labeled cooler comparison. Spatial NPZ/maps, original full-region reports and
separate original/clean ROI analyses are retained for every block.

## Current handoff after the replug/retry branch

Fresh `firmware/post-cooldown-retry-inventory/`, **18:19:03–18:19:05 UTC**,
verified **20,000/20,000 µs, auto-exposure 0, gain 24**. Firmware version hash
`2b2b9908915acd9df465690cd6804211d11b6bdc2c89cf4167b8555f4fd78644` and
depth-intrinsics hash
`fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf`
match the fresh replug inventory. Final CPU/NPU readings were **59.8/59.5°C**.
The two-second cached-sysfs witness at 18:21:56–18:21:58 records the exact
serial stable at USB device 97, 480 Mb/s, with unchanged `auto` power policy.
No capture is left running; the video nodes have no open owner.

Diagnostic changes add first-fault UVC/V4L2 tracing, cached USB identity/power
observation, a closed lifecycle experiment harness, hash-pinned offline ARM
inspection, and the opt-in thermal quiet boundary. The post-run failure-path
review additionally blocks UVC after V4L2 transport faults and requires the
quiet boundary before restoration on an ordinary capture exception. These
extra error paths are mock-tested, not physically fault-injected.

**108 targeted tests passed** across stationary quality, lifecycle, FPS/view
probes, validation reports and thermal comparison. Selected Ruff checks passed;
mypy passed for the lifecycle and thermal harnesses. This is not a full-repo
test claim. No production GUI/GPU/tracking code, live meshing, persistent
settings, USB power/driver policy or firmware was changed in this continuation.

`source-snapshot-lifecycle-run/` preserves the source used for the successful
hardware retry, before the final failure-path review.
`source-snapshot-lifecycle-final/` preserves the reviewed diagnostics, tests and
updated documents. `evidence-manifest-lifecycle-retry.json` records size/SHA256
for the diagnostic root without rewriting earlier captures or manifests; file
integrity is not a physical-scene or accuracy certificate.

**14 ms / gain 24 at approximately 6.6 fps remains the provisional stationary
finalist**, not a universal depth-preserving maximum or a handheld fix. The
firmware flash branch still lacks an authenticated exact-model image, complete
device-specific preservation and demonstrated recovery. The next operator-
dependent branch is controlled turntable motion at a fixed repeatable speed;
independently timed handheld motion remains separate. Further thermal accuracy
qualification needs a stable fixture and repeatable thermal history. Broader
working-distance/lighting tests and the full 15/30-minute candidate sequence
are pending. The complete original plan is not marked finished.

## P3: fresh low-speed start and framing dependency, 18:41–18:54 UTC

The operator started the turntable at the requested lowest steady setting and
warned that moving it to reach the underside could shift its placement. This
is a **new setup**. No prior pixel masks, physical alignment or fitted rotation
axis were transferred. Neither physical RPM nor rotation direction was measured.

New evidence root: `/home/ben/scans/diagnostics/20260907-motion-low/`.

`placement-check/` uses the tested 64-frame 14 ms settled-post lifecycle
procedure. All 64 frames were complete and unique, zero short/sequence losses,
6.534 fps including short-window/startup phase. Peak CPU/NPU 59.4/58.6°C.
Original 20/20 ms, auto 0, gain 24 was restored and the returned USB identity
settled. `placement-inspection/` plots clearly show rotation. The 64-frame
largest-component bounds have at least two pixels of top margin and no border
contact, but do not cover all later views. Its multi-frame reference/scene
counts are inspection aids for moving data, **not stationary quality metrics**.

`baseline-a/`, ending **18:49:51 UTC**, then captured **320 complete unique
frames at 4.506513 fps**, with zero short frames or sequence gaps. This was
an acquisition-only 20/20 ms / gain 24 baseline; all raw payloads and actual
delivery timestamps were saved after stream closure. CPU/NPU peaked at
**60.2/60.0°C**. The quiet-stop boundary, fresh original timing/gain readbacks,
and final returned USB observation all passed without a returned UVC transport
fault. This is an acquisition/lifecycle result, not a tracking result.

### The longer rotation exposes clipping

`baseline-a-framing/` retains per-frame bounds, hashes and full-frame plots
without temporal averaging. The main connected region reaches the **top
sensor row in 98/320 frames**, contiguous indices **18–115**. Inspection of
`0080-depth.png` and the other selected views confirms the tallest target
feature reaches the top boundary; this is not solely support/background contact.
There are also 27 frames with connected returns at the bottom, chiefly support.
The total 121 border-contact frames must not all be described as clipped target
top frames. No missing component was observed.

The first 64-frame check was insufficient to establish whole-rotation framing.
The predeclared scene gate therefore pauses the baseline/candidate motion
matrix. `baseline-a/quality-validity.json` explicitly excludes this capture
from quality qualification while preserving its acquisition/lifecycle evidence.
No 300-frame candidate, GPU tracking replay or motion-speed pass is claimed.
No new firmware flash, tracking algorithm/gate or meshing change was made.

### Prepared diagnostics and final state

The existing acquisition helper now has explicit **320/640/1200-delivery
motion budgets**, capped at 180 seconds, while ordinary probes retain their
2–256-frame/90-second bounds. Only the settled-post lifecycle mode accepts
motion timing presets 20/14/10 ms; other modes retain their old short behavior.
Five-second thermal monitoring and first-fault/no-retry protection are unchanged.
Raw files, SHA256, delivery indices and monotonic timestamps are preserved after
shutdown, including partial complete payloads on a capture exception. A frame
budget is a ceiling, not a claim that every requested frame arrived.

Offline motion replay now accepts this raw evidence format with an explicit
common depth band and at least 12 complete frames/one second settling excluded.
It checks hashes, lengths, timestamps, restored capture status and validity
annotations before GPU work. It retains production preparation/tracking rules;
the new raw-format hardware replay was **not executed on this flagged scene**.
The independent revisit audit can report actual delivery-time separation.

**135 targeted tests passed**, including motion budget/mode restrictions,
partial raw preservation after stop, timing/hash/scene validity, existing
thermal/lifecycle behavior and framing with empty/border-contact inputs.
Selected Ruff and mypy checks passed; no full-repository regression claim.
`source-snapshot/` and `evidence-manifest.json` preserve reviewed diagnostics
and this branch's evidence without rewriting prior diagnostic roots.

Fresh no-stream `final-state/`, **18:54:22–18:54:29 UTC**, verifies original
**20,000/20,000 µs, auto-exposure 0, gain 24**, CPU/NPU **59.4/58.6°C**, and
stable USB identity after descriptor closure. No capture is left running;
the turntable itself remains under operator control. Device logs, including
stop-path warnings, remain in the report; successful host readbacks are not
proof that the firmware emitted no warnings.

Next physical step: slightly raise/tilt the scanner upward so the part sits
lower in its image, leaving visible space above the tallest feature throughout
rotation. Keep the current turntable setting. After that adjustment, acquire
a **new** framing/baseline sequence; do not merge the clipped setup with it.

## P3: reframed diagnostics, 19:03–19:16 UTC

Root: `/home/ben/scans/diagnostics/20260907-motion-reframed/`. This is a separate
physical setup after the operator's upward adjustment, not a continuation of
the earlier clipped-top data.

| Capture | Complete / unique | Observed fps | Short / sequence missing | Peak CPU / NPU |
| --- | --- | --- | --- | --- |
| Original 20 ms / gain 24 | 320 / 320 | 4.506611 | 0 / 0 | 60.7 / 60.0°C |
| Diagnostic 14 ms / gain 24 | 320 / 320 | 6.585824 | 0 / 0 | 61.1 / 60.4°C |

Both captures completed the quiet-stop boundary, freshly verified the original
20/20 ms, auto 0, gain 24 after restoration, and ended with stable returned USB
identity and no returned transport fault. These are acquisition/lifecycle
results, not whole-part geometry qualification.

Full-frame inspection shows that the top adjustment worked: minimum top margins
are 52 and 53 pixels respectively, with zero top-border contact. The bottom is
now too crowded. The cleaner 14 ms plots `0080-depth.png`, `0160-depth.png` and
`0319-depth.png` show solid base corners meeting the lower boundary. Connected
returns touch the bottom in 133 baseline and 114 candidate frames, but those
counts include support and are **not exact semantic target-clipping counts**.
Both capture roots have explicit qualification-exclusion annotations. No longer
capture or candidate replay was taken in this placement.

Before the lower-edge inspection was resolved, `baseline-a-replay/` ran the
unchanged production GPU-assisted tracker on 300 complete inputs after 12
settling frames, common 60..300 nominal-mm band, stride 1. The AMD RX 9070 XT
FP64 path was reported active, with SciPy correspondence search. **300/300
frames were accepted**, with no loss stop, in 28.973 seconds wall time. The
independent image-period check on the 66.356-second input prefix found no
credible revisit; geometric coherence is **not checkable**, not passed.
The replay was concurrent with diagnostic acquisition and is not an isolated
host ceiling or a live-GUI fps measurement. Its original report, decisions,
poses, timings and hashes remain unchanged; a sidecar records the subsequent
framing exclusion. Acceptance does not establish tracking correctness.

Fresh no-stream `final-state/`, **19:16:27–19:16:34 UTC**, verifies original
20,000/20,000 µs, auto 0, gain 24 and a stable USB identity, with final CPU/NPU
60.2/59.5°C. No stream was started by that inspection. The targeted six-file
diagnostic test set was rerun: **135 passed in 1.80 seconds**. No production
GUI/GPU/tracking code, firmware, power policy or experimental mesh changed.
Selected diagnostic source snapshots and a file-integrity manifest preserve
this branch; they are not a complete checkout or a geometry certificate.

At 19:17 UTC the operator reported using the app preview to move the scanner
so that the bottom is no longer cut off. A fresh video-node ownership check
found no camera owner. Continue with new placement evidence under
`/home/ben/scans/diagnostics/20260907-motion-centered/`, using the separately
declared longer bounded baseline to inspect framing and seek image revisits.

## P3: centered placement and visible-geometry diagnostic

The operator used the app preview to improve lower framing; subsequent camera
access was free. Root: `/home/ben/scans/diagnostics/20260907-motion-centered/`.
`baseline-a/`, **19:19:13–19:21:54 UTC**, completed **640 complete unique
20 ms frames at 4.511398 fps**, zero short frames/sequence gaps. Peak reported
CPU/NPU was **62.0/61.3°C**. Original timing/gain restoration, the quiet-stop
boundary and final returned USB stability passed without a transport fault.

`baseline-a-framing/` measures at least 50 pixels above the main component and
no top contact. Some views have clear lower target margin, while a base corner
meets the last row in selected views (`0160-depth.png` has three component
pixels on that row). The 308 total bottom-contact frames include support and
must not all be labeled clipped-target frames. The exact amount of missing
target geometry is not established. Whole-part margin/coverage remains
unqualified in `baseline-a/quality-validity.json`.

To avoid making complete framing a prerequisite for every useful tracking
diagnosis, a narrower branch was declared **before replay**: inspect gross
drift of the observed geometry without qualifying full-part coverage. The
explicit diagnostic-scene mode propagates source warnings and a false quality-
qualification flag into top-level, per-stride and revisit-audit reports.
Normal replay still refuses flagged scenes; hash, capture/restoration, settling,
timestamp and geometric gates are unchanged. No production tracking changes.

**136 targeted tests passed in 2.24 seconds**. Selected Ruff checks passed.
An initial mypy dictionary-key inference error was fixed with an explicit
type annotation; mypy then passed for both modified offline tools. The
GPU-assisted 628-frame settled baseline replay was started with the diagnostic
flag; results follow when complete. This is not live-GUI throughput testing.

The intermediate no-stream `final-state/`, **19:24:34–19:24:42 UTC**, verified
original 20/20 ms, auto 0, gain 24 and stable USB identity, with last reported
CPU/NPU **60.7/60.0°C**. Despite its directory name, this is an intermediate
state record if further hardware capture follows; later captures retain their
own restoration records.

### Baseline tracking result and faster acquisition

The centered baseline accepted **628/628** settled frames in 63.662 seconds
wall time, with RX 9070 XT FP64 normals active. Preparation totaled 11.398 s,
tracking 49.302 s; neither is a GUI-display timing. The input span is 138.983 s.
Three credible image pairs are separated by **115.068–115.072 seconds**, with
raw median matched-pixel differences 0.045–0.080 nominal mm. These are image-
derived revisit intervals, not independently measured motor RPM.

The legacy fixed-lag check scores 97.17% at worst, and the individually matched
audit scores **98.79%** at worst within the unchanged **1.5 nominal-mm** gate.
This is good observed-geometry consistency, not a metrology accuracy claim.
The same saved input at strides 2/3/4 accepts every offered frame (314/210/157)
with weakest individually matched agreement **99.05/99.16/98.95%**. These
larger between-frame jumps do not reproduce faster within-pattern motion.
All reports retain the false full-quality qualification flag.

`candidate-a/`, **19:29:18–19:32:39 UTC**, then completed the 180-second
acquisition ceiling at **6.598159 fps**, saving **1,183 complete unique
frames**, zero short/sequence losses. The requested 1,200-frame maximum was
not reached because of the declared time limit, not an acquisition fault.
Peak reported CPU/NPU was **63.3/62.6°C**; original 20/20 ms, auto 0, gain 24
was restored and the returned USB epoch remained stable without a transport
fault. Full plots retain major target features; 167 bottom-connected frames
include support and are not exact clipped-target counts. The same observed-
geometry-only qualification warning is attached before replay.

`cadence-summary.json` retains interpolated p50/p95/p99/max delivery gaps from
all complete-frame timestamps. Baseline p95/p99 is **383.99/385.62 ms**;
candidate **245.79/247.11 ms**. Alternating short/long intervals remain present.
The original summary's upper-middle definition of median differs from an
interpolated median for an even number of gaps; both evidence and convention
are retained, and the mean-based fps is unaffected. No production median
implementation was changed in this diagnostic work.

Candidate GPU replay was started on all **1,171** settled inputs at strides
1/2/3/4, with unchanged preparation/tracking/geometric gates and an explicit
diagnostic-only label. Results follow when complete.

### Candidate drift persists despite full acceptance

All candidate replay strides completed without a tracking-loss stop. Every
offered frame was accepted, but **all four fail** the unchanged 75% geometric
agreement gate on individually image-matched revisits:

| Stride | Accepted / offered | Legacy fixed-lag minimum | Individually matched minimum |
| --- | --- | --- | --- |
| 1 | 1171 / 1171 | 67.99% | **64.63%** |
| 2 | 586 / 586 | 46.94% | **52.80%** |
| 3 | 391 / 391 | 54.47% | **69.72%** |
| 4 | 293 / 293 | 23.26% | **33.07%** |

Artifacts: `candidate-a-replay/` and `candidate-a-revisit.json`. The full
four-stride process took 267.400 s. At stride 1, preparation/tracking totaled
22.281/97.882 s for 1171 frames: about **9.75 processed frames/s**, excluding
GUI, saving, coherence analysis and other overhead. The RX 9070 XT FP64 path
was reported active in every stride. This supports processing headroom over
6.60 fps in this offline workload, not a measured live-GUI ceiling.

### Same raw pair, different registration history

`candidate-a-pair-investigation/` examines the worst actual stride-1 pair,
replay source indices **195/948**, raw deliveries **207/960**. Their raw median
matched-pixel difference is **0.070 nominal mm**, separated by 114.029 s.
The original tracked relative transform differs by about 0.784 degrees and
has 1.247 nominal-mm translation magnitude; these are transform quantities,
not a separately calibrated physical error measurement.

Using identical poses for the two nearly repeated views gives **99.657%**
projected agreement; direct two-frame point-to-plane alignment gives
**99.657%**, 0.2505 nominal-mm fitted residual, all source points finding
correspondences before the default 60% trimming. This fit is a diagnostic
contrast, **not held-out validation or a correction applied to the scan**.
The original accumulated poses still score **64.632%**. Full images, raw
hashes, transform conventions and original/direct comparisons are preserved.

Cross-capture image matching finds the closest original baseline view at
source index **280** (raw 292), raw median difference **0.055 nominal mm**.
Its next approximately 519-frame revisit lies beyond the first baseline's
628-frame replay. Therefore the shorter baseline's good score did **not**
test this troublesome angle twice; it cannot by itself isolate an exposure
effect from different angle coverage or registration history.

`candidate-a-phase-replay/` starts the unchanged candidate byte sequence at
original replay index **195**, retaining all **976** remaining frames in their
original order and at their actual recorded timestamps. Only the diagnostic
starting-view offset changes. The original failed trajectory is not rewritten.
All 976 offered frames are accepted in 95.843 s. The automatic period sampler
returns **not checkable** for this shifted window; that result is preserved.

The separately preselected exact pair becomes indices **0/753** in that replay.
`candidate-a-phase-pair.json` verifies that paths, hashes, delivery indices and
timestamps exactly match the prior pair. Under the new accumulated poses it
scores **99.714%**, versus **64.632%** originally, without any direct pair fit
being injected into the tracker. This demonstrates dependence on registration
history for that pair. It does not establish that the shifted whole trajectory
is correct, erase the original failure, or isolate one specific internal
keyframe/normal/solver mechanism.

The source has bounded, correlated references and explicitly retains the first
keyframe. That makes anchor retention and independent revisit validation useful
next investigation targets, **not proven sole causes or changes made here**.
Simply raising fps or accepting more frames is not a demonstrated fix for this
failure. No mesh or production tracking policy was changed.

### Longer original-speed control and current conclusion

`baseline-b/`, **19:43:04–19:46:23 UTC**, uses the same 180-second acquisition
ceiling as the candidate: **809 complete unique frames at 4.516541 fps**,
zero short frames/sequence gaps, peak CPU/NPU **62.5/61.7°C**. Original timing
and gain restoration and the returned USB stability check passed without a
transport fault. The main component has at least 49 pixels of top clearance;
332 bottom-contact frames include support, so full-part coverage remains
explicitly unqualified. No framing warning was removed to run this control.

Its 797-frame settled GPU replay completes in 78.543 s, with every offered
frame accepted. Preparation/tracking totals 14.524/60.543 s. All five sampled
image pairs are credible, separated by **113.802–113.804 seconds**. The legacy
minimum is **86.65%**, independent matched minimum **82.58%**, passing the 75%
gate **for these sampled observed surfaces**. This is less margin than the
shorter baseline. The starting view differs from the candidate; equal duration
does not turn this into a repeated matched-phase causal exposure experiment.
The image-derived intervals also vary across runs (initial baseline ~115.07 s,
candidate ~114.0–114.3 s, later baseline ~113.80 s). No perfectly constant
motor speed or independent physical RPM is asserted.

The bounded diagnostic block is complete. **14 ms / gain 24 remains a
6.6 fps provisional stationary finalist, not a reliable qualified moving-scan
profile.** These captures establish no acquisition losses at that rate, while
the offline tracker processes the candidate at about 9.75 frames/s excluding
GUI/IO/coherence overhead. They also reproduce silent geometric error that
depends on the registration's history. The exact same failing pair improves
when starting there; faster USB, more accepted frames or more GPU throughput
alone is not a demonstrated remedy.

Next engineering tests are exact-byte CPU/GPU parity on this newly failing
sequence, followed by isolated reference-retention/independent-revisit
experiments with the original failures retained as regression cases. This is
a proposal for the next diagnostic/fix block, not a production fix made here.
Full-part framing, three matched-phase baseline/candidate pairs, higher
physical turntable speeds and independently timed handheld motion remain
pending. Earlier thermal depth-shift and firmware-recovery gates still apply;
the complete original validation plan is **not marked finished**.

Final no-stream `final-state-after-controls/`, **19:51:47–19:51:54 UTC**,
verifies **20,000/20,000 µs, auto-exposure 0, gain 24**, CPU/NPU **60.2/59.5°C**,
and stable USB identity. The video nodes have no open owner after completion.
The turntable remains under operator control. No firmware, persistent device
settings, USB power/driver policy, experimental meshing or production GUI/
tracking code changed in this block.

**137 targeted diagnostic tests passed in 2.67 seconds** after adding the
explicit starting-view offset; selected Ruff and two-tool mypy checks passed.
The offset requires diagnostic mode, retains source hashes/timestamps, cannot
replace the declared settling exclusion and cannot enable quality qualification.
`source-snapshot-start/`, `source-snapshot-diagnostic-replay/`,
`source-snapshot-phase-replay/` and `source-snapshot-final/` preserve the
selected source versions and final documentation. The local closed pair
investigation scripts are also preserved beside their outputs. These are
selected snapshots, not a complete environment/checkout archive.
`evidence-manifest.json` is the final size/SHA256 inventory of this root;
integrity does not certify scene completeness or measurement correctness.

## Reference-history reliability tests, from 20:00 UTC

Evidence root: `/home/ben/scans/diagnostics/20260907-tracking-reliability/`.
Earlier roots remain sealed. These experiments use real saved bytes, not
synthetic scan geometry. Full-part framing caveats remain in every report.

The exact 1,171-frame candidate failure is **identical across CPU reference,
original GPU and fresh GPU repeat**: all inputs, preparation point counts,
complete decisions and poses agree exactly. All 1,171 frames are accepted,
but actual matched-revisit agreement is only 64.632%. Processing-only rates
are CPU 7.742 fps, original GPU 9.745 fps, fresh GPU 10.646 fps, excluding
GUI/save/coherence costs. GPU arithmetic is not the distinct cause of this
reproduced failure. `parity-comparison.json` records exact comparisons/hashes.

Two separately predeclared reference ablations leave acceptance/geometry
thresholds untouched. Disabling only the merged-model candidate fails badly:
0% worst agreement despite every frame being accepted. Raising only retained
keyframes/clouds from 12/32 to **64/96** improves the same four credible revisit
pairs to **95.183%** worst agreement. At source 948, the keyframe candidate
now uses old source 194, where the default had discarded that viewpoint.
This supports useful-history retention as a mechanism, not just faster input.

The 64-view experiment exposed an exact-result cache too small for its >4,000
directed pose pairs. Increasing that bounded cache from 2,048 to 16,384 entries
changes **no complete decision, pose, reference selection or policy value**
on a second full replay. Processing improves **7.864→9.960 fps** (26.66%).
This cache-only change is the first production-code change in this block;
the live default still retains 12 keyframes. Its bound/exactness regression
test passes. `cache-comparison.json` preserves the full equality checks.

A new dense audit reuses the existing image matcher at 25 positions, verifies
every participating raw hash, and retains the original sparse/fixed-lag results.
On the development recording, 22 credible pairs give **58.321% default versus
90.203% retention64** worst agreement. On the longer original-speed control,
the dense result improves **79.393%→89.623%**. These sampled observed-geometry
checks pass the unchanged 75%-within-1.5-nominal-mm gate under retention64;
they do not establish calibrated dimensional accuracy or complete coverage.
Additional saved-recording and fresh physical checks are in progress.

Idle inspection at 20:16 UTC verifies original 20,000/20,000 µs, auto exposure
0 and gain 24, no transport fault, and CPU/NPU at most 59.4/58.6°C. No mesh,
firmware, USB power-policy or persistent scanner-setting change was made.

### Longer validation rejects the retention-only proposals

The favorable short development/control results did **not** generalize:

| Recording | Default, dense | 64 references, dense | Stable 64-reference archive, dense |
| --- | ---: | ---: | ---: |
| Development, 1,171 frames | 58.321% | 90.203% | Not evaluated |
| Original-speed control A, 628 | 98.786% | 98.997% | Not evaluated |
| Original-speed control B, 797 | 79.393% | 89.623% | Not evaluated |
| Earlier moving recording, 2,703 | 6.400% | 0.981% | 12.814% |
| Fresh six-minute recording, 2,359 after settling | 76.126% | 30.752% | 64.878% |

**No retention or candidate-preference experiment has been promoted to the
application.** All gates remain 75% within 1.5 nominal mm. The stable archive
keeps recent clouds separately and discards the newer member of a closest
reference pair; it also fails. These failures are retained rather than choosing
the earlier favorable recordings as validation.

The first fresh acquisition, **20:30:02–20:36:11 UTC**, delivered 2,371 unique
complete frames at **6.598247 fps**, no short frames or sequence gaps, under its
360-second ceiling. The 2,400 number was a budget, not its delivered count.
Peak CPU/NPU **64.2/63.5 C**; original 20,000/20,000 µs, auto 0 and gain 24
were verified restored after a settled quiet boundary, with stable returned
USB identity and no transport fault. Framing remains diagnostic-only: ≥50 pixels
top margin, no absent main component, but 272 frames touch a border with the
combined part/support component. No semantic full-part coverage is asserted.
A second 360-second capture reserved for later physical confirmation also
completed **2,371 unique complete frames at 6.598194 fps**, with no transport
fault and verified restoration, peak **64.2/63.9 C**. Its trajectory has not
been used to choose a profile.

### Two independently reproduced correctness defects

1. **Final consistency false pass.** On the first fresh retention64 trajectory,
   the original fixed-lag check passes at 84.203%, and its five actual matched
   pairs pass at 99.541%, while the 25-position actual-pair check finds 30.752%.
   The shared production `check_coherence` now uses 25 image positions and
   projects every credible best-image pair. The median lag is descriptive,
   not a substitute for those pairs. Coherence report schema 2 records selection,
   sample count and unprojectable pairs; incomplete projection evidence cannot
   yield a clean pass. The old reports remain untouched. A synthetic regression
   reproduces the missed bad pair, and `final-gate-fresh-retention64.json`
   verifies the corrected false verdict on the exact original saved poses.
   This is a qualification fix, not a repaired trajectory.

2. **Rotation roundoff feedback.** The keyframe-first diagnostic crashes at
   source frame 35 on the CPU reference, matching the GPU rotation error.
   The saved partial trajectory's orthogonality defect grows from ~3e-15 at
   frame 5 to 3.60e-7 at frame 34. A scanner/GPU-independent stationary pose
   `p.compose(p.inverse().compose(p))` fails after just 20 iterations. The
   transpose-based coordinate round trip amplifies tiny non-orthogonality.
   Composition now performs one polar correction only for tiny defects in
   the band above 1e-12 and at most 1e-10, **after the unchanged 1e-6 validation guard**.
   Larger defects are not repaired, and scale/reflection refusals remain.
   The same stationary reproducer completes 1,000 iterations with maximum
   defect 7.06e-13 and unintended point displacement below 9e-12 coordinate
   units. All 98 geometry tests pass. CPU and GPU now both accept the same
   first 100 real frames with exactly identical decisions and poses.
   `repaired-prefix-parity.json` records all exact comparisons and report
   hashes. This prefix has no credible image period and cannot qualify motion
   quality. Full keyframe-first outcomes below still fail; no general tracking
   fix is claimed.

Before these last two source changes, **450 core tests passed, 2 skipped**;
the diagnostic suite then passed **167 tests**, and the later preference
selector passed all **41 report-tool tests**. The new coherence regressions
pass **11 tests, 1 skipped** (synthetic sweep shorter than the default minimum
revisit lag). A broader post-fix regression run is in progress. Source snapshots,
the before/after round-trip reproducer and failed partial CPU replay are retained.

### The remaining alignment variants also fail

The repaired keyframe-first mode accepts 2,702/2,703 older frames but gives only
**2.416%** minimum dense revisit agreement. It accepts all 2,359 first-fresh
frames yet scores **0%**. Fixing the floating-point crash does not make that
preference reliable; it remains an offline diagnostic only.

Two final predeclared alignment changes keep the production 12/32 retention,
original candidate preference, all gates and identical raw order:

| Single fitting change | Development, 1,171 frames | First fresh, 2,359 frames |
| --- | ---: | ---: |
| Keep all ICP correspondences (0.8→1.0) | 63.982% | 72.082% |
| Existing point-to-point fitting, CPU | 2.405% | 11.875% |

Each accepts every offered frame and fails the unchanged 75% consistency gate.
No retention, eviction, preference or fitting default has changed. Point-to-point
explicitly disables normal estimation and requires CPU mode; the helper refuses
a contradictory GPU label before reading inputs. Shared-load replay timings
are not throughput benchmarks. This block does not establish that arbitrary
parameter tuning or a firmware flash would fix registration.

Post-fix core tests: **570 passed, 2 skipped** (geometry, tracking/recovery/loss,
capture, preparation, ICP and live acceleration). GUI/export/project regressions:
**100 passed**. The final diagnostic/coherence run: **194 passed, 1 skipped**,
including missing-evidence verdict precedence and early invalid-mode refusal.
These suites overlap in coherence coverage; the counts must not simply be added
as unique tests. Selected Ruff and four-file mypy checks pass. The final check
is used by the app's reconstruction/final-consistency path, not as a newly
implemented per-frame live drift detector.

The reserved second physical recording was replayed once with production
default policy and only the tested correctness/cache changes.
Its full framing audit has no absent main component and at least 50 pixels of
top margin; 271 frames have a border-touching combined part/support component.
The sidecar explicitly preserves diagnostic-only/full-part coverage limitations.
No profile selection is made from this confirmation trajectory.

Final no-stream inspection, **21:35:46–21:35:54 UTC**, verifies exact serial,
20,000/20,000 µs, auto-exposure 0 and gain 24, with no transport fault and CPU/NPU
at most **60.2/59.5 C**. The hardware tests made no persistent setting, firmware,
USB power/driver-policy or experimental-mesh change. The turntable remains
under operator control.

### Reserved confirmation: still not reliably qualified

`confirm-default/report.json` completes all **2,359/2,359 accepted frames**
without a stop, but the final 25-pair actual-image check is **INCOHERENT**:
**73.5528%** minimum agreement, below the unchanged 75% requirement. All
25 pairs are projectable. The weakest pair is accepted/source indices
491 and 1,235 (raw deliveries 503 and 1,247), with median projected offset
−0.7060 nominal mm. Neither this near miss nor the first fresh default's
76.126% is evidence of a repeatable result with healthy margin.

`confirmation-pair-control.json` reopens and hash-checks the exact two inputs,
loads the recorded poses without repair or registration, and reproduces
**73.5528% exactly**. Projecting those same raw views in unchanged camera
coordinates gives **99.9469%**. That is a diagnostic pair-level control, not a
replacement trajectory, alignment search, full-part quality pass or firmware
inference. It supports the conclusion that the accumulated registration is
responsible for this pair's disagreement, despite complete frame delivery.
The confirmation recording was not searched for a better tracking profile.

The final additional geometry regression covers both origin-centred and
translated stationary coordinate round trips: **99 tests pass**. The production
tracking overview now correctly describes its unchanged model→adjacent→keyframe
preference; the earlier wording disagreed with the actual implementation.
No experimental mesh was constructed. Video nodes have no open owner after
the final device inspection.

This engineering block is complete; the larger reliability goal is **not**.
Retained production changes are limited to the bounded exact pose-pair cache,
guarded tiny-roundoff correction on transform composition, and the broader
actual-pair final consistency check. None of the failed diagnostic tracking
profiles is enabled in the app. Acquisition at 14 ms/gain 24 remains a
**6.6 fps provisional candidate**, not a qualified reliable moving-scan mode.
The ~10 fps offline processing evidence excludes GUI, saving and the final
consistency audit; it is not a new sustained live-GUI claim.

The next registration work must use these adverse saved-byte cases, not only
the initially favorable development segment. Original handheld performance
still needs an operator-assisted, independently timed motion recording; a
rotating part cannot reproduce all handheld translation/rotation modes or
within-pattern motion. Full-part/support coverage, repeated matched-phase
quality pairs and long-duration candidate thermal/depth-drift qualification
also remain open. No maximum safe handheld speed has been established.

`source-snapshot-final/` preserves the selected final production, diagnostic,
test and documentation sources. `evidence-manifest.json` is the final bounded
file-size/SHA256 inventory of this root, excluding itself. These are selected
source snapshots and file-integrity evidence, not a complete environment
archive or certification of scan correctness. Earlier evidence roots remain
unchanged.
