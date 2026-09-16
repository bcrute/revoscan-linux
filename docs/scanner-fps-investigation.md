# MINI acquisition timing investigation — 2026-09-06/07

The physical MINI **can deliver substantially more than 4.5 fps**. Its existing
firmware accepts volatile frame-time and exposure controls that the ordinary
V4L2 frame-interval request does not exercise effectively. No firmware image,
calibration, projector current, sensor gain, CPU clock, or production default
was changed. Experimental live meshing remains disabled.

Unit: `A22704522A6F00A64`, firmware `v2.8.7.20220610`, algorithm
`v7.5.7.0426`, USB `2207:110c`, direct host port `1-1` at **480 Mb/s**.
Depth mode is 640 × 400 Y16, 512,000 bytes per complete frame. The higher
measured rates below also used that same USB 2 link.

Later [thorough validation](scanner-validation-results.md) identifies **14 ms /
gain 24, approximately 6.6 fps**, as a provisional stationary clean-reference
finalist across three views. A five-minute retry delivered 1,975 complete unique
frames without short/sequence losses and restored state after a six-second
post-stop quiet boundary. Runtime controller logs and pinned ARM disassembly
confirm routine delayed gadget teardown/reopening; the host workaround is
diagnostic-only and does not prove the earlier unrecoverable failure repaired.
Warm matched-pixel shifts partly reverse on cooling, so temperature-invariant
accuracy, moving geometry and handheld speed remain unqualified. The original
baseline comparisons still fail; faster raw rates below are not useful-depth
or tracking guarantees.

## Acquisition-only baseline

`tools/scanner_fps_probe.py` captures into RAM without tracking, rendering, or
raw-frame disk writes. Kernel buffer timestamps and host delivery intervals
are recorded. Hashing and image-difference analysis happen after stream close.

| Host buffers | Poll interval | Deliveries | Complete / unique | Measured fps | Sequence drops |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 ms | 96 | 96 / 96 | 4.482 | 0 |
| 2 | 1 ms | 96 | 95 / 95 | 4.523 | 0 |
| 8 | 1 ms | 96 | 95 / 95 | 4.516 | 0 |
| 4 | 10 ms | 96 | 95 / 95 | 4.519 | 0 |

The small differences are consistent with finite-window/startup phase, not an
improvement. Typically one incomplete startup delivery occurs on rapid stream
restarts. Short deliveries are counted but never hashed or sent to tracking.
The earlier 15 → 10 → 5 → 15 fps standard V4L2 test also stayed at 4.475–4.477
actual fps; see [tracking-performance.md](tracking-performance.md).

The device's own DSP log independently shows left/right processing starting
every approximately **443 ms**, followed by `send 0 z16 depth` and
`send 1 z16 depth` approximately 60 ms apart. This explains the host's
alternating approximately 63/380 ms gaps. These are two distinct payloads,
not duplicated host deliveries. Baseline useful depth payload throughput is
only about 2.3 MB/s; a bus trace was not needed to establish the timing control's
effect, and no USB 3 renegotiation occurred.

## Firmware evidence and exact controls

The running `/tmp/zxmain` controller was recovered with the previously known
bounded vendor file-read protocol. It is 166,412 bytes, SHA-256
`decdc3ad4e34449cefee8543c790866cfc87dd9279a873c3abab124b9754b0f1`, identical to
the historical controller artifact. It was disassembled offline with Capstone
5.0.9 and pyelftools 0.33 installed only in `/tmp/scanner-fps-analysis-deps`.

Relevant ARM virtual addresses in that exact binary:

| Address | Meaning established from instructions / subsequent measurement |
| --- | --- |
| `0x21bf4` | Builds `{3, 0x911, value}`: set depth exposure |
| `0x21c3c` | Builds `{3, 0x910, value}`: set frame time |
| `0x21c60` | Reads exposure through `0x9b1` |
| `0x21c80` | Reads frame time through `0x9b0` |
| `0x21c8c` | Reads auto-exposure mode through `0x9b2` |
| `0x25384` | Device `process_set_depth_fps_val` handler; 640-wide presets include 15,000 and 7,000 µs |
| `0x255b0` | Loads the 7,000 µs preset (`movw r5, #0x1b58`) |
| `0x190fc` | Standard depth absolute-exposure handler calls the exposure setter |

The UVC commit path computes nominal timing fields, but the recovered direct
frame-time setter's call site is in the separate device FPS handler. Live tests,
not just disassembly, establish that ordinary S_PARM is ineffective here and
the two explicit timing controls work. The malformed standard exposure range
(maximum zero despite current value 20,000) was not patched or trusted.

Verified transport uses **UVCIOC_CTRL_QUERY on the existing depth V4L2 node**:
unit 4, selector 7, SET_CUR, 60-byte NUL-padded ASCII
`echo s 0x910 VALUE >/dev/rk_preisp` or
`echo s 0x911 VALUE >/dev/rk_preisp`. No driver detachment is needed for these
tests. The Linux query structure is `unit, selector, query, size, pointer`
(`@BBBHP`, 16 bytes on this host), not `unit, selector, size, query, pointer`.
[Kernel UVC documentation](https://docs.kernel.org/userspace-api/media/drivers/uvcvideo.html)
describes that ABI.

Every candidate was checked against **fresh, timestamped DSP getter replies**,
not USB transfer success alone. Fresh logs name both setters and the device's
projector/sensor timing adjustment. Requested and device-reported quantized
sensor times differ slightly:

| Requested frame time | Logged sensor frame time | Logged projector value |
| ---: | ---: | ---: |
| 20,000 µs | 19,865 µs | `0x1f` |
| 15,000 µs | 14,738 µs | `0x17` |
| 10,000 µs | 9,612 µs | `0x0f` |
| 7,000 µs | 6,408 µs | `0x0a` |

These projector values are observed firmware outputs, not guessed current
register writes. Auto-exposure remained zero. To shorten timing, exposure is
reduced before frame time; restoration increases frame time before exposure.
Only recovered finite presets are allowed. No command-ID sweep was performed.

The vendor's [SDK header](https://github.com/Revopoint/3DViewer/blob/516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7/thirdparty/3DCamera/include/hpp/Types.hpp)
also describes exposure in microseconds and its impact on frame rate, but the
unit's controller and fresh DSP replies are the stronger evidence for the
specific wire commands. Headers for other firmware/models are not sufficient.

## A/B/A speed and usable-geometry checks

First 20 → 15 → 20 ms experiment, 80 deliveries each: **4.478 → 6.042 →
4.516 fps**. A second 128-delivery matrix measured **4.492 → 9.069 → 13.036
→ 4.526 fps** at 20 → 10 → 7 → 20 ms. All complete payloads were distinct;
zero sequence drops. Each rapidly restarted run after the first had one short
startup delivery.

An independent matrix retained 96 complete frames per setting, sampled
temperatures during acquisition, then ran ordinary preparation and SciPy
tracking **offline**, after each raw stream closed:

| Frame time / exposure | Raw fps | Valid-code pixel fraction | Tracking accepted | Median tracking time |
| ---: | ---: | ---: | ---: | ---: |
| 20 / 20 ms | 4.483 | 29.13% | 96 / 96 | 121.4 ms |
| 15 / 15 ms | 5.995 | 27.75% | 96 / 96 | 117.4 ms |
| 10 / 10 ms | 8.994 | 27.96% | 96 / 96 | 117.2 ms |
| 7 / 7 ms | 12.949 | 26.03% | 96 / 96 | 108.6 ms |
| 20 / 20 ms restored | 4.482 | 29.83% | 96 / 96 | 120.5 ms |

All five runs had zero sequence drops, zero short deliveries, unique payloads,
and no coarse recovery in replay. Valid-code fraction means `0 < code < 65535`,
not a metrological acceptance test. The turntable's angular phase was not held
fixed between runs, so differences in pixel yield mix viewpoint and exposure
effects. The separate 128-frame 7 ms run had a lower 21.07% yield, against
28.87–29.06% baseline. **Do not claim equal geometry quality at 7 ms.** The
10 ms setting is a strong next integration candidate, not yet a production
default or a proven handheld-accuracy fix.

## Full GUI: acquisition is no longer the only bottleneck

The visible GPU-normal-estimation harness at 10 ms captured
`/home/ben/scans/scan/20260906-212319`: 180 deliveries, 179 complete frames,
**179 accepted**, no tracking loss, one short delivery and nine reported
sequence drops. Delivered rate was **5.138 fps**, median tracking 130.2 ms.
There were 365 real GPU eigensolve calls. The CPU still owns the remaining
preparation, tracking, model refresh and orchestration. This is not 9 fps
end-to-end performance. Internal/backpressure losses need not all appear in
the host's UVC sequence counter.

The first GUI attempt failed before capture during startup/driver handback;
the depth USB function remained present but had no capture node. The existing
bounded interface-0 version-read lifecycle successfully restored the V4L2
nodes; 20 ms timing was then explicitly restored and verified. No USB reset,
power cycle, firmware write, or root escalation was used. The successful
harness reused the freshly verified version string instead of detaching the
driver again on GUI startup. This is an isolated harness adjustment, not a
production startup fix. The failed run's restoration wrapper also had a
double-close error; it was corrected before the successful rerun. Failed
measurements are not included as throughput successes.

## Temperatures

Read on the scanner, not the PC: `/sys/class/thermal/thermal_zone0` is
`cpu-thermal`; zone 1 is `npu-thermal`. Temperatures are returned in millidegrees
Celsius. The short quality matrix measured CPU 60.2–62.5°C and NPU 60.0–61.3°C.
The full GUI test stayed at CPU ≤62.5°C / NPU ≤61.7°C.

The scanner's CPU thermal configuration reports passive trip points **75°C and
85°C**, and critical **115°C**, using `power_allocator`. Zone 1 returned no
content for those same trip-point files. That is not an established independent
NPU limit. A 90°C sensor reading would already be above both CPU passive trip
points; critical shutdown is not a recommended continuous operating target.

The later test policy stops at **80°C**, fails closed on missing temperature
telemetry, and leaves all firmware thermal protections unchanged. This policy
is not a manufacturer safety rating. These sensors do not establish projector,
DSP, lens, case, or ambient temperature. The MINI's
[official guide](https://download.revopoint3d.com/support/download/mini/mini-quickstartguide-en-v3.0-2023.7.27.pdf)
specifies a 0–40°C operating environment; that is not an internal silicon
junction-temperature limit. Long-run dimensional drift is still unmeasured.

### Five-minute maximum-tested-rate soak

`tools/scanner_thermal_soak.py` held one depth stream open for 300 seconds at
7 ms/7 ms, without tracking/rendering limiting the scanner. It hashed every
complete payload, sampled temperature every five seconds and pixel yield
approximately once per second. The run finished on its planned time limit:

- **12.987 fps**, 3,891 complete frames, **3,891 distinct payloads**.
- **Zero sequence drops, zero short frames**, no stream stall.
- Five consecutive minute windows stayed at approximately **12.98–12.99 fps**;
  sampled mean valid-code fractions ranged from 23.16% to 23.83%, with no
  monotonic collapse as temperatures rose. This does not measure geometric drift.
- CPU started at 61.1°C and finished at **67.3°C**; NPU started at 60.4°C and
  finished at **66.6°C**. Both stayed below the first observed CPU passive trip.
- Device timing was still 7,000/7,000 µs after the stream; **20,000/20,000 µs
  was restored and verified** afterward. Auto-exposure remained zero.

| Approximate elapsed time | CPU | NPU |
| ---: | ---: | ---: |
| Before scan | 61.1°C | 60.4°C |
| 1 minute | 64.2°C | 63.5°C |
| 2 minutes | 65.1°C | 64.8°C |
| 3 minutes | 66.0°C | 65.3°C |
| 4 minutes | 66.9°C | 65.7°C |
| After 5 minutes | 67.3°C | 66.6°C |

The temperature rise slowed but did not demonstrate a true indefinite plateau.
This is a five-minute maximum-tested-output acquisition soak, not an hour-long
test, a full-GUI tracking endurance test, or a dimensional-drift measurement.

## Reproduction and evidence

Run from this checkout with no other scanner application open:

```bash
.venv/bin/python tools/scanner_fps_probe.py --frames 96 --output /tmp/new-fps-baseline.json
.venv/bin/python tools/scanner_timing_probe.py --execute --quality --frames 96 \
  --timings 20000 15000 10000 7000 20000 --output /tmp/new-timing-quality
.venv/bin/python tools/scanner_thermal_soak.py --execute --output /tmp/new-thermal-soak.json
```

Reports refuse existing destinations. The timing probe verifies the original
20 ms/20 ms/manual baseline before writing and restores it in `finally` under
a termination shield. It also checks sensor temperatures every five seconds.
SIGKILL, physical disconnection and system failure cannot be software-cleaned;
failed restoration is an error, never reported as success. Device observations
create only uniquely named small `/tmp` RAM log excerpts; no persistent device
files are written and logs are not cleared.

Evidence from this investigation:

Preservation update, 2026-09-09: these listed temporary artifacts, the named
controller report and timing-quality logs now have hash-verified durable copies
under `/home/ben/scans/diagnostics/20260909-historical-throughput-preservation/`.
See `source-inventory.json` for the original-to-copy mapping. Original paths
and report bytes remain unchanged; this archival action is not a new scan.

- `/tmp/scanner-acquisition-baseline-20260906.json`
- `/tmp/scanner-fps-kernel-20260907/kernel.txt`
- `/tmp/scanner-fps-controller-20260907/controller.bin` and `report.json`
- `/tmp/scanner-timing-aba-20260907/report.json`
- `/tmp/scanner-timing-fast-20260907/report.json`
- `/tmp/scanner-timing-quality-20260907/report.json` and per-stage kernel logs
- `/tmp/scanner-live-gpu-20260906-212319.json`
- `/tmp/scanner-faster-live-gpu-b-20260907.json`
- `/tmp/scanner-thermal-soak-7ms-20260907.json` (five-minute soak)

Capstone/pyelftools are not required to use these probes. No production module,
driver, system permission or dependency was changed in this investigation.

Verification: **313 offline tests passed** across the new probe checks,
frame-interval probe, stream lifecycle, ownership, preparation and tracking.
Ruff passed for the four new tools and their tests; mypy passed for all four
tools with `MYPYPATH=src:tools --follow-imports=silent`. `git diff --check`
passed. A final 32-frame baseline smoke test delivered 32 complete frames with
zero short deliveries or sequence drops; a subsequent fresh getter check again
confirmed 20,000/20,000 µs and auto-exposure zero. Scanner sessions were closed
after testing. No commit or push was made.
