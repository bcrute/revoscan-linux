# Reliability follow-up: independent object face B

8 September 2026, following the operator's repositioning confirmation at about
14:33 UTC. This continues [the roadmap](reliability-roadmap.md) and
[the registration/thermal record](reliability-registration-continuation.md).

**Result: this face is substantially more stable, but moving reliability is
still unqualified.** Production code and all frozen experimental algorithms
remain unchanged. S2/S3's previous moving-control rejections remain binding.

Evidence: `/home/ben/scans/diagnostics/20260908-reliability-view-b/`.
The acquisition, fixed static comparisons and subsequent 48-fit directional
audit each have a written protocol preceding their execution. Previous sealed
evidence is read-only. This root's final `evidence-manifest.json` records file
integrity, not a release verdict.

## Acquisition and scene

The exact MINI `A22704522A6F00A64` was unowned before the run. The established
settled-post lifecycle verified 20/20 ms, gain 24 and auto exposure off, retained
the quiet stop boundary, restored and checked settings, and verified final idle.

- Run: 14:34:33–14:36:02 UTC, including setup and shutdown observations.
- All **320 deliveries complete and distinct**, no detected sequence drops,
  short frames or transport faults; all raw frames and host delivery times kept.
- Observed **4.50627 fps**. Delivery intervals remain uneven: approximately
  57–386 ms, rather than a uniform 222 ms. This is not a 10 fps mode.
- Sixteen chip observations; maximum/final **60.2 / 60.0 C**. These are scanner
  CPU/NPU readings, not certified case/optics temperatures.
- Full images at 0, 160 and 319 show a clearly different face, exposing the
  slotted side below the inclined plate. The view appears stationary. Fragmented
  lower/background returns remain visible and are not cropped away.

There is no external pose encoder. Identity is a fixed-camera diagnostic, not
calibrated ground truth. No camera/scene adjustment was requested during capture.

## All-frame frozen replays

All three preparations offered and accepted all 320 frames without terminal
loss. All used the existing 60..300 band, 900-point target, default tracker and
verified AMD RX 9070 XT FP64 GPU normals with SciPy correspondences. No settling
prefix was excluded, no pose was clamped, and no hindsight transform aligned
the results to identity. The same 20 fixed raw checks were applied in each run.

| Preparation | Face A maximum apparent surface motion | Face B maximum | Face B weakest placed raw check |
| --- | --- | --- | --- |
| Production compacted stride | 27.3894 nominal mm | **0.6909** | **93.8444%** |
| Frozen S2 pixel rank | 0.1960 | **0.05544** | **99.7801%** |
| Frozen S3 hierarchical grid | 0.4242 | **0.06444** | **99.7801%** |

Face B's raw identity checks never fall below **99.7801%**. Production's worst
same-pair loss after applying its poses is 5.9359 percentage points. Both frozen
samplers preserve the identity check scores at all 20 checked pairs. This is
stronger static evidence for sample stability, not a moving-control pass or a
CAD accuracy measurement. Timings from concurrent offline replays do not
qualify complete live GUI latency.

Across all 319 adjacent pairs, median retained locations are:

- Valid raw pixels: **98.6268%**.
- Production sampled pixels: **1.9956%**.
- S2 sampled pixels: **98.4462%**.
- S3 sampled pixels: **98.3352%**.

Production still replaces almost its entire sampled pixel set on each delivery.
But this face does not develop the large drift seen on face A, so turnover
alone is not a sufficient explanation of the scene-dependent failure.

## D5: directional constraints, not a new threshold

Eight predeclared adjacent pairs per view (ending at 1, 40, 80, 120, 160, 200,
240 and 319), each with three preparations: **48 unchanged-solver fits**.
Every fit and information calculation completed. No support plane was removed
in any of these preparation instances. All fits start at identity and therefore
do not recreate the full tracker's model/reference history.

For production preparation, median normalized local condition is **95.10** on
face A versus **38.56** on B; median weakest normal-direction eigenvalue is
**0.01496** versus **0.04120**. These local diagnostics are consistent with B
providing stronger directional constraints. Median apparent adjacent movement
is 0.638 versus 0.209 nominal mm.

However, S3 on face A has a *worse* median condition of **268.57**, while its
median adjacent apparent movement is only **0.02363 mm**. A condition-number
cutoff alone therefore cannot explain or safely classify these results. The
matrices depend on estimated normals/correspondences; their residual-based
uncertainty is not a calibrated probability. No new acceptance gate, normal
weight, calibration correction, sampling order or target budget was introduced.

## Verification and disposition

The selected lifecycle, capture, registration-audit and frozen-sampler suite
passed **97 tests**, no failures/errors/skips. These complement the prior full
suite; they are not a second full-suite run. Direct comparisons confirm that
production preparation/tracking/ICP/GPU normals and both experimental samplers
are unchanged from the previously sealed source. New files are diagnostic
adapters, evidence and documentation; unrelated owner changes are preserved.

The scanner is idle with baseline settings verified. No mesh, firmware flash,
driver/dependency change, commit, push or deployment was performed.

Next: a new controlled-motion **diagnostic**, not release qualification. Start
the turntable at its slowest setting with the camera fixed and the whole object
visible, then report ready. The protocol in `NEXT-MOTION-DIAGNOSTIC.md` calls
for a bounded 640-delivery baseline recording, all raw evidence retained, and
the existing frozen replay/check procedure. A static improvement cannot erase
the known moving failures. Handheld release trials remain deferred until an
eligible correction and its regression gates actually pass.
