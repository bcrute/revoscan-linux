# Saved signals and timing: six longer/handheld captures

Read-only audit, 2026-09-09 05:53–06:01 UTC. Evidence is under
`/home/ben/scans/diagnostics/20260909-reliability-normal-taper/` in
`stage-b-channel-inventory.json`, `stage-b-image-provenance.json` and
`stage-b-timing-audit.json`, with their executing audit scripts. No capture,
timestamp, fit, raw score, gate, production setting or calibration is changed.

The six original Stage B capture directories (or the immediate capture parent
when raw files occupy a `raw/` subdirectory) contain all 8,333 required inputs.
They also retain 36 original warmup frames outside that qualification roster.
All 8,369 named depth files are 512,000 bytes. This inventory does not rehash
those raw files or replace the already sealed raw-input verification.

Nine possible image files exist, all in slow D. Every pixel in every image is
exactly reproduced by applying the existing depth renderer to its corresponding
saved raw depth frame. No independent RGB/IR stream is identified in these six
directories. The earlier separate Stage A audit found the same limitation for
its fifteen inspection images. This is bounded filesystem/schema evidence,
not a universal binary-content detector or proof of hardware incapability.

The deliberately broad metadata-key scan also matches `minimum_top_margin_pixels`
because its name contains `imu`; that is a scene-margin field, not inertial data.
The three JSONL traces contain bounded vendor-control/stream lifecycle records
(only 60-byte transfers when a transfer size is present), not a recorded image
or inertial sensor stream. No hidden intensity channel is inferred from depth.

## Kernel and host timing are not interchangeable

All 5,139 selected inputs in the three instrumented captures map exactly to
their original raw hash, delivery index and host time. Their kernel timestamps
are labelled `TIMESTAMP_MONOTONIC` and `TSTAMP_SRC_SOE`. Those are recorded
driver labels, not independent verification of physical exposure semantics.
Both clocks increase strictly, but their alternating intervals have opposite
phases. Medians below are milliseconds, grouped by the destination delivery
index parity across every adjacent pair in each fixed original input prefix.

| Capture | Kernel even / odd interval | Host even / odd interval |
| --- | ---: | ---: |
| Slow D, 20 ms | 382.828 / 60.058 | 59.844 / 382.810 |
| Development extended, 14 ms | 245.358 / 57.665 | 57.578 / 245.228 |
| Later adverse | 243.818 / 59.285 | 59.247 / 243.854 |

Median dequeue-minus-kernel delay is 65.897 / 63.170 / 66.296 ms respectively;
maxima are 388.830 / 250.817 / 249.583 ms. Saved host times follow the recorded
dequeue instant by median 1.034 / 1.053 / 1.067 microseconds. Thus the audit
confirms the host timestamp's relation to dequeue, not the sensor exposure.
No phase shift, clock calibration, smoothing or interpolation is fitted.

The older adverse and combined handheld project files contain 2,703 and 303
relative `frame_times_s` values, respectively, but the legacy replay's original
input manifest does not ingest them. Paused handheld has no such array. No
per-buffer kernel timing is available in these three legacy capture folders.
Adding missing catalogue fields would not itself make the tracker time-aware.

The actual `IncrementalTracker.offer` API accepts a cloud and optional adjacent
fit, not a timestamp. Its prior extrapolates a measured transform per offered
frame. The replay stores modern host times as evidence, not a motion-model
input. A time-aware prior would therefore require a separately declared and
tested algorithm and validated timing semantics. Existing F1/M1/M7 audits also
show no actual prior seed or prediction-allowance-dependent acceptance in those
twenty particular trajectories; timestamp substitution is not a demonstrated
repair for their observed errors. Handheld fallback behavior is a separate case.

## What this supports next

An independent color/IR or inertial alignment experiment cannot be honestly
replayed on channels these recordings do not contain. New synchronized channel
capture and calibration would be needed before making that claim. Likewise,
the timing discrepancy motivates validation of what the device/driver labels
actually mean, not using one clock as truth merely because its label is SOE.
The previous stationary delivery-parity audit found weak cross-view agreement
of apparent depth offsets; neither audit justifies a depth-phase correction.

These are additional diagnostic leads, not evidence that every depth-only
method must fail. The predeclared M24/M25 tests and any survivor qualification
retain priority. No package, firmware, scanner or external-system action occurs.
