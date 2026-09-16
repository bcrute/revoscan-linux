# Live GPU throughput — 2026-09-06–07 follow-up

Later validation on 2026-09-07 is recorded in
[scanner-validation-results.md](scanner-validation-results.md): an additional
packed-covariance optimization improves the controlled offline replay by 11.2%,
and a short awake-display stationary run applies every complete depth preview
at ~10.4 fps. Crucially, revisiting the earlier moving capture finds substantial
geometric drift despite all poses being accepted. The historical rates below
remain measurements, **not evidence of correct accumulated shape**.

## Outcome

GPU FP64 normal estimation is now integrated into the GUI capture worker.
Install `.[gpu]`; **GPU normals** in the technical toolbar defaults on for
auto/SciPy. SciPy still handles correspondence search. Initialization checks
128 deterministic eigensolves against the CPU reference and explicitly reports
CPU fallback if initialization or dispatch fails. The accelerator/cache are
worker-local, not global patches; relocalization workers keep their CPU path.
Experimental live meshing remains disabled. Scanner timing defaults are unchanged.

After the operator reconnected the scanner, the visible GUI reached **9.76 fps**
at 8 ms, accepting **179/179 complete delivered frames**, with one short delivery
and 10 sequence gaps. A repeat with nearest-neighbor index caching measured
9.78 fps with the same counts. These improve on the preceding 7.86 fps physical
run and 5.14 fps prototype. The index-cache run needed its benchmark process
terminated after capture because display-worker shutdown lingered; scanner
baseline restoration still verified. Cleanup was corrected and regression-tested.

The subsequent **five-minute visible GUI scan sustained 9.01 fps**, accepting
**2,703/2,703 complete frames** without tracking loss. It had one short delivery
and **307 sequence-reported missing deliveries**. All 2,703 depth previews and
676 model snapshots were applied; 1,559 colour pictures were shown. The process
exited normally and fresh 20/20 ms restoration verified. **10 fps sustained
end-to-end has not been demonstrated.** All live tests in this report used the
turntable; they do not establish a maximum reliable handheld speed.

**These faster exposures are not depth-preserving defaults.** The new stationary
A/B/A test found substantial missing returns (below), which the earlier moving
comparison underestimated. Scanner timing defaults remain 20/20 ms.

Accumulated-model updates remain approximately 2–2.5 Hz, not 10 Hz. Raw
acquisition, delivered/tracked frames, and model refreshes are distinct.

## Implementation and parity

- GPU covariance preparation retains CPU `math.fsum` centroids and sequential
  covariance sums. The shader keeps the tested FP64 division refinements.
- NumPy batches ICP normal-equation products with sequential accumulation;
  the original 6×6 solve, residuals, convergence and tracking gates are unchanged.
- Indexed overlap uses the original inclusive, correctly rounded distance
  test on proposed neighbours, including points exactly on the gate.
- Packed depth rendering avoids per-pixel Python byte objects and repeated
  code-tuple conversions. A histogram replaces repeated sorting; advisory
  consumers share a read-only uint16 view. Pixels/statistics remain identical.
- Bounded, value-keyed caches reuse placed coverage observations and keyframe
  pose gaps. Batched angular comparisons shortlist contenders and recompute
  winning dot products with the reference arithmetic.
- Support-plane refinement batches projections without changing operation
  order or inclusive band comparisons. No filtering or thinning policy changed.
- Support covariance batches independent products, retaining `math.fsum` for
  every reduction. A one-target, per-backend index cache avoids rebuilding the
  same small SciPy tree on each ICP iteration; target values, not list identity,
  invalidate it. Large targets are not retained.
- Ordinary coverage rebuilds use one spawned helper process, with at most one
  immutable request containing 40 sampled clouds. Capture never waits for its
  result. Pause/recovery boundaries discard obsolete jobs; Stop discards all
  outstanding display work. The first and final pictures remain synchronous.
  Normal cleanup uses nonwaiting shutdown so the bounded current job can finish
  its result-pipe write; forcibly terminating it caused an observed exit hang.

The stored 179-frame capture `20260906-212319` reproduced **exactly the same
complete decision records and poses** as the pre-change CPU reference after
the main optimizations. Preparation/tracking/coverage replay took 21.636 s
(8.27 processing fps, median tracking 75.4 ms), excluding acquisition and GUI
drawing. This is not a live-rate claim. Subsequent pose-gap memoization is an
additional exact-result cache.

The final production GPU preparation/tracking replay after index and support
covariance changes again matched all **179 complete reference decision records
and poses exactly**. It took 17.360 s, excluding acquisition, model building and
GUI drawing; it is not a physical throughput measurement.

Packed rendering alone measured 5.0 ms versus 22.4 ms for the tuple renderer
on a recorded frame (30 warm iterations). Coverage built in 115.5 ms cold,
then about 57 ms with cached observations. Changing sampled views means live
captures do not get the all-cache-hit workload.

## Acquisition and quality

Fresh device readback confirmed these settings; all raw frames below were
complete and unique, with zero sequence gaps:

| Exposure / frame-time request | Raw fps | Frames |
| --- | ---: | ---: |
| 20 / 20 ms | 4.491 | 128 |
| 9 / 9 ms | 9.597 | 128 |
| 8 / 8 ms | 11.031 | 128 |

The 8 ms setting is a candidate for feeding a 10 fps pipeline, not an adopted
GUI default. Timing transitions now read the current frame window and order
writes so a longer exposure never precedes a longer window.

The turntable was not confirmed stopped. Exploratory nearest-view comparison
against the baseline found median return retention of **95.8% at 9 ms** and
**94.9% at 8 ms** in the baseline object's eroded interior. Median mutual depth
differences were 0.130 and 0.177 mm. This is **not proof that meaningful detail
is preserved**: rotation, occlusion, viewpoint mismatch, temporal noise and
excluding edges from the metric all limit the conclusion. A stationary A/B/A
test including thin features and edges was still needed at that point.

`tools/depth_quality_compare.py` records each match/residual and these limitations.
The bounded timing probe's `--save-raw` writes complete frames only after
streaming ends, so disk writes do not throttle its acquisition measurement.

### Stationary A/B/A — completed after reconnect

The operator confirmed the turntable stopped. One uninterrupted stream tested
20, 10, 9, 8, 7 and 20 ms; each transition discarded 12 complete frames and at
least one second before retaining 40 full-resolution, unique depth frames.
Every timing and final restoration earned fresh device-side readback. The
largest connected baseline return region supplied a fixed 80,595-pixel mask;
the interior erodes three pixels, and the edges are reported separately.

| Timing | Raw fps | Interior returns | Edge returns | Persistently lost stable pixels |
| --- | ---: | ---: | ---: | ---: |
| 20 ms, initial | 4.60 | 99.72% | 91.98% | 0% |
| 10 ms | 8.90 | 84.19% | 56.44% | 17.12% |
| 9 ms | 9.76 | 83.89% | 56.24% | 17.30% |
| 8 ms | 10.96 | 82.57% | 52.34% | 18.58% |
| 7 ms | 13.05 | 77.37% | 46.04% | 24.33% |
| 20 ms, final | 4.44 | 99.75% | 92.64% | 0% |

“Stable” means present in at least 90% of the 80 baseline frames (76,742
pixels); “persistently lost” means present in fewer than 10% of candidate
frames. Interior/edge return fractions average all corresponding pixels and
frames. This mask can include the support surface; it is not semantic object
segmentation. Missing baseline-visible surfaces are not dismissed as noise.

Using the repository's nominal depth conversion, the 90th-percentile absolute
change between the two baseline pixel medians was 0.045 mm. Candidate median
depth departures at mutually observed stable pixels were about 0.20–0.22 mm
at the 90th percentile. Temporal MAD at 8 ms was 0.030/0.070 mm (median/90th
percentile), versus 0.0225/0.0525 mm initially. These measure repeatability and
relative changes, **not calibrated absolute accuracy**. The missing returns
are the stronger reason not to promote 7–10 ms as equivalent-quality modes.

This completed stationary comparison supersedes the exploratory moving
retention figures above. There is currently no demonstrated setting that both
feeds 10 fps and preserves baseline depth on this scene. Faster live runs are
processing/thermal stress tests, not evidence that the exposure tradeoff vanished.

## USB interruption and temperature

Successful GUI runs restored 20,000/20,000 µs with fresh verification. Later
raw sweeps repeatedly disconnected at a subsequent transition. Two recovered,
but at **22:00:46 CDT** the scanner disconnected and could not enumerate again.
Host logs show descriptor timeouts, protocol errors (-71), and an unsuccessful
**host-initiated** port power-cycle attempt. These tests issued no reset or
firmware-flash command. The cause of the disconnection is not established.

The final raw test's last sample was **61.1°C CPU / 60.8°C NPU**; the earlier
GUI run peaked at 62.5/61.7°C. The 80°C test cutoff was not reached. These
sensors do not measure the projector/optics or establish whole-device safety.

The last sweep **could not verify restoration**: its descriptor became stale,
and a fresh attempt found no scanner on USB. Restoration now always reacquires
the serial-resolved node, but cannot write while enumeration is absent. Hardware
commands were stopped and a physical reconnect requested. After reconnect,
verify timing/temperature and restore baseline before resuming bounded tests.

The operator subsequently reconnected the scanner. Fresh checks confirmed
20/20 ms and auto-exposure zero before testing resumed. The single-stream
stationary sweep completed without disconnection; CPU/NPU peaked at 59.8/59.1°C.
The first asynchronous-coverage live run peaked at 60.2/59.5°C and verified
20/20 ms restoration afterward. The earlier failed sweep's unverified state
is historical, not the current device-state claim.

The five-minute 8 ms GUI run began at 58.5/57.7°C and peaked at **64.7/64.4°C**
(CPU/NPU), with five-second telemetry and an 80°C stop policy. The cutoff was
not reached, and baseline restoration ended at 64.7/63.9°C. This is not an
indefinite thermal-plateau or projector-temperature qualification. Median
tracking was 80.4 ms; the final part slowed considerably despite no tracking
refusals. Other CPU-consuming jobs were observed on the PC afterward, so this
was not an isolated-machine benchmark and the slowdown's cause is unresolved.

An earlier attempted soak collected no scan frames because of argument-name
shadowing in the temporary duration-enabled benchmark wrapper. It was stopped,
baseline restoration verified, and the wrapper corrected before the completed
five-minute run. It is not counted as acquisition or thermal-soak evidence.

## Local evidence

Preservation update, 2026-09-09: the specifically listed temporary artifacts
were copied without changing their original bytes to
`/home/ben/scans/diagnostics/20260909-historical-throughput-preservation/`.
Its `source-inventory.json` maps original paths to durable copies; all selected
paths were present. The sealed copy is integrity-verified, not a new validation
of the historical performance or quality claims below.

- `/tmp/scanner-before-gui-optimization-20260907.json`: reference decisions.
- `/tmp/scanner-after-solve-overlap-20260907.json`: unchanged decision records.
- `/tmp/scanner-live-production-20260906-215612.json`: 7.86 fps GUI run/stage timings.
- `/tmp/scanner-production-packed-20260907.json`: its thermal/restoration record.
- `/tmp/scanner-production-replay-20260907.json`: offline pipeline replay.
- `/tmp/scanner-quality-candidates-20260907/`: baseline raw set/failed transition.
- `/tmp/scanner-quality-candidates-b-20260907/`: 9 ms raw set/failed transition.
- `/tmp/scanner-quality-candidates-c-20260907/`: 8 ms raw set/failed restoration.
- `/tmp/scanner-depth-quality-matched-20260907.json`: exploratory quality comparison.
- `/tmp/scanner-stationary-quality-20260907-a/`: six stationary raw sets/readbacks.
- `/tmp/scanner-stationary-quality-analysis-20260907.json`: matched-pixel findings.
- `/tmp/scanner-live-production-20260907-000111.json`: 9.76 fps asynchronous live run.
- `/tmp/scanner-production-async-coverage-8ms-20260907.json`: thermal/restoration record.
- `/tmp/scanner-live-production-20260907-000803.json`: 9.78 fps, followed by exit hang.
- `/tmp/scanner-production-cached-index-8ms-20260907.json`: termination and verified restore.
- `/tmp/scanner-final-parity-20260907.json`: final exact 179-frame production replay.
- `/tmp/scanner-live-production-20260907-001402.json`: completed five-minute GUI run.
- `/tmp/scanner-production-five-minute-8ms-b-20260907.json`: thermal/readback/clean exit.

These `/tmp` artifacts are experimental evidence, not durable test fixtures.
`tests/test_live_acceleration.py` covers exact rendering, solve/refinement/
coverage parity, inclusive overlap, worker-local cache lifecycle and CPU fallback.
Real GPU tests require `SCANNER3D_TEST_GPU=1` and never open the scanner.
`tests/test_async_coverage.py` checks bounded requests, stale results, real
capture completion/Stop/loss boundaries, spawned snapshot parity and process exit.

Final targeted regression: **369 passed, 12 skipped** across processing,
capture, GUI lifecycle and scanner-probe tests. A separate real-GPU run plus
the new async/quality tests passed **59 tests**, including CPU/GPU numerical
parity and helper failure fallback. Ruff passed on the touched implementation
and tests; mypy passed on eight selected source/tool/test files. No firmware
flash, projector-current change, commit, push or deployment was performed.
