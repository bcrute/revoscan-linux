# Pocket visibility and fixed-view exposure check — 2026-09-09

## Outcome

The pocket is not simply invisible to this scanner. Reducing exposure from
20 to 10 ms repeatedly recovers a coherent part of its floor and largely removes
the missing-return stripes on the surrounding top. Returning to 20 ms brings
the missing returns back. All **120 short-exposure frames across two views**
support a direct floor-to-top measurement without multi-view tracking, CAD
placement or scaling. The second view follows a physical reconnect; it is not
an unchanged-pose repeat of the first view.

Primary pocket medians are **11.974 / 11.977 / 11.975 mm**, against the existing
owner reading **12.10 mm**. Different nearby plateau patches move the estimate
by up to about 0.08 mm in the second view. This is useful raw-geometry evidence,
not a calibrated accuracy claim or a resolved global scale discrepancy.

**Surface-label correction:** the old region called `base` is on the turntable,
not the printed base. Its short-exposure losses must not be reported as lost
printed-base coverage. Two correctly identified printed-base patches retain
100% of their points in all 200 second-view frames. At 10 ms, their plane RMS
improves from roughly 0.09--0.11 to 0.025--0.030 mm; primary plateau RMS improves
from roughly 0.52 to 0.039 mm. This makes 10 ms a promising candidate for a
fully framed capture, not a global preset for other faces/ranges/materials.

The first acquisition completed but its final control read lost USB after the
wrapper omitted the documented quiet-stop boundary. The additive guarded
adapter has now completed one hardware run with full post-quiet baseline
readback and stable post-close identity. Both failures and successes remain
recorded. No firmware, production, calibration, mesh or multi-exposure fusion
changes were made. Next physical dependency: reframe the whole target, with
the camera supported and turntable stopped, before a short full-view check.

## Current-view evidence, not a premature failure conclusion

The operator reported that RGB could see into the pocket while depth could
barely do so, then clarified that the camera had been moving and requested a
current check before recording a failure. The earlier GUI screenshot is only
a preview, not a scan. No conclusion that the pocket is unscannable was adopted.

`/home/ben/scans/diagnostics/20260909-pocket-visibility-TaYFBR/current-view/`
contains the requested fresh check, 19:52:25--19:52:35 UTC: 32 complete distinct
raw frames, zero drops/shorts, unchanged 20/20 ms, manual mode 0, gain 24 and
CPU/NPU maxima 60.2/59.5 C. First/last common-pixel absolute depth differences
are median 0.035 mm, p90 0.140 mm, p99 0.595 mm; 99.574% are within 1 mm.
This is a stable-view observation, not dimensional or tracking qualification.

The raw renderer shows zero codes as missing without a GUI depth-band filter.
Fixed manual image regions were chosen from this view before exposure changes:

| Region | Image bounds x0,y0,x1,y1 (exclusive) | Pixels |
| --- | --- | ---: |
| Apparent pocket centre | 128,261,152,280 | 456 |
| Surrounding plateau | 146,151,196,201 | 2,500 |
| Turntable control (legacy JSON key `base`) | 225,285,275,335 | 2,500 |

The initial view returned only 1--4 pocket-centre pixels per frame, none present
in all 32 frames. Plateau/turntable returned roughly 57%/97%. These are selected
image-space patch statistics, not exact semantic area coverage of entire faces.

## One-variable exposure comparison

Evidence root (closure identity is recorded below):
`/home/ben/scans/diagnostics/20260909-pocket-exposure-Fe9kV9/`.

Keep the scene fixed, frame period at 20 ms, manual exposure mode 0 and gain
register 24. The sequence is 20/15/20/10/20 ms exposure. The existing stationary
harness retains 40 complete full-resolution frames after at least 12 discarded
frames and one second following every setting transition. All five stages
complete: **266 full deliveries, zero sequence drops or short frames, 200 settled
frames saved**. Every transition's timing/gain and every stage's ending gain
verify. CPU/NPU maximum is **61.1/60.4 C**; this is not projector temperature.

| Exposure stage | Pocket patch returns | Plateau patch returns | Turntable control returns | Frames supporting pocket measurement |
| --- | ---: | ---: | ---: | ---: |
| 20 ms, initial | 0.70% | 56.86% | 97.34% | 0/40 |
| 15 ms | 13.04% | 99.83% | 92.56% | 0/40 |
| 20 ms, middle | 0.64% | 56.61% | 97.48% | 0/40 |
| 10 ms | 58.81% | 100.00% | 69.42% | 40/40 |
| 20 ms, final | 0.76% | 56.44% | 97.39% | 0/40 |

At 10 ms the pocket patch returns 265--272 pixels per frame, with 263 present
in all 40 frames. A coherent partial floor is visible in
`exposure-a/stage-3-exposure-10000.png`; remaining parts remain unknown.

This establishes exposure sensitivity in the current fixed view, consistent
with the earlier [stationary results](scanner-validation-results.md). Saturation
or nonlinear fringe decoding remains a mechanism hypothesis, not a direct
measurement of raw fringe intensities. It is not a new independent proof that
the sensor is faulty or that all missing points are caused by exposure.

Shorter exposure also changes measured depth: on initially stable plateau pixels,
the 10 ms versus initial 20 ms median-map absolute change has p90 0.938 mm.
The final 20 ms bracket itself differs by p90 0.380 mm on its common stable
plateau pixels; the striped baseline is not calibrated truth. On the turntable patch,
corresponding p90 changes are 0.060/0.055 mm, but 10 ms persistently loses 462
of 2,394 initially stable pixels. This control is not on the printed base, so
it does not establish a printed-base coverage penalty. Neither visibility nor
repeatability establishes an absolute bias correction.

## Direct pocket measurement and support sensitivity

Use unchanged device-derived Pl intrinsics and depth scale. Each frame fits
independent TLS planes to the fixed floor and plateau patches, with fixed
three-MAD trimming, at least 200 retained points, at least 2 mm span in both
principal tangent directions, and a planar-support check. No temporal median,
CAD pose, nominal plane spacing, forced parallelism or caliper value enters
the fit. Distance is plateau-normal separation at the retained floor centroid.

All 40 frames at 10 ms pass those support checks; all 160 other frames are
explicitly refused for insufficient floor support. The 10 ms result:

- Separation min/median/max: **11.966807 / 11.974368 / 11.983055 mm**.
- Median difference from the owner's 12.10 mm reading: **-0.125632 mm**.
- Retained floor points: 254--266; representative principal spans 3.86/4.52 mm.
- Floor plane RMS median 0.05461 mm; plateau RMS median 0.03549 mm.
- Independently fitted plane angle: 0.641--1.354 degrees, median 0.983 degrees.
- Across all fitted supported floor footprints, local separations range
  11.93219--12.03059 mm. Surface nonparallelism is reported, not flattened away.

The plateau fit extends to the pocket location. A declared post-primary
sensitivity check repeats all 40 frames using two additional visible plateau
interiors, keeping the floor patch and fitting rules fixed:

| Plateau patch | Separation min / median / max (mm) |
| --- | --- |
| Primary | 11.966807 / 11.974368 / 11.983055 |
| Near-upper | 11.994066 / 12.011954 / 12.024101 |
| Left | 11.946245 / 11.958530 / 11.974311 |

The primary reproduces exactly. Every alternative is retained; none is selected
for closeness to 12.10 mm. Patch dependence and unknown caliper/print uncertainty
preclude treating the tiny within-block spread as accuracy. The observed local
depth discrepancy already exists without inter-frame tracking, but does not
uniquely determine a global scale fault: 0.7% of 12.10 mm is just 0.0847 mm, and
this is a small, partial, somewhat nonparallel surface pair.

## First USB lifecycle failure and additive correction

The last in-stream stage verified original 20/20 ms and gain 24. The legacy
inner harness reports completion/restoration, but the outer wrapper then failed
during its additional final gain read with EPROTO followed by ENODEV. Kernel
records at 20:01:20 UTC show disconnect of device 21, unsuccessful enumeration
attempts 22--25 and failed kernel-initiated port recovery. The scanner remained
absent on subsequent read-only checks. **Full post-stop restoration is not
verified** and the overall first-block lifecycle is failed, not cleanly completed.
No additional controls, software reset or hub/driver/power-policy write followed.

`run_check.py` retains the failed wrapper. It omitted the existing six-second
quiet shutdown/re-enumeration boundary and made duplicate final reads. The new
`quiet_capture.py`/`run_check_quiet.py` use the previously tested boundary and
first-transport-fault UVC guard, then resolve the device afresh for one final
state/gain/temperature check and a post-close identity observation. No production
files were changed. Its first real hardware use is the second-view block below;
one successful lifecycle does not establish general lifecycle reliability.

Tests: the existing stationary suite passes 46 cases. Eight new geometry cases
pass before actual measurement. Five additional lifecycle cases cover clean
completion, mid-candidate interruption/restoration, transport failure, an
unsatisfied quiet boundary and failed final readback. The combined new suite
passes 13 cases. Its first run's five test-fixture setup errors and correction
are preserved; they did not access the scanner. These are 59 distinct software
tests, not 59 hardware trials or scanner qualification.

## Reconnect block: second view and repeat within that view

The owner confirmed reconnect. Preflight at 20:23:55 UTC verified the exact
serial, original 20/20 ms/manual 0/gain 24, and CPU/NPU 58.0/57.7 C. The declared
**20/10/20/10/20 ms** sequence completed with 266 full deliveries, zero drops or
shorts, 200 settled raw frames and observed whole-run cadence 4.541 fps. Frame
period remained 20 ms; this is an exposure-quality test, not a 10 fps claim.
Maximum CPU/NPU was **60.2/59.5 C**. Final settings/gain/temperatures were read
once through a fresh descriptor after the six-second quiet boundary. The device
automatically re-enumerated 26 to 27 during that boundary; no manual reconnect,
software reset or power-policy write was needed. Wire trace has no transport
fault, final baseline/gain verify, and two-second post-close identity is stable.

The mandatory framing comparison found that physical handling shifted the scene
roughly 30 pixels at the pocket. The old floor ROI includes plateau now. Same-
pixel first-block/new-baseline depth differences (median 1.805, p90 8.850 mm)
are therefore a framing mismatch, not sensor drift. `repeat-framing.png` retains
the old boxes and both views. No unchanged-ROI pocket analysis was run on the
new capture. `SECOND-VIEW-PROTOCOL.md` fixes a replacement floor ROI
**[158,263,182,282]** before calculating new separations. Its selection is
post-capture visual inspection, not an independent preregistered accuracy test.
Original primary plateau and legacy turntable-control bounds remain, but do
not identify exactly matched physical patches across views.

| Stage | Exposure | Pocket returns | Plateau returns | Turntable returns | Pocket measurement |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 20 ms | 2.81% | 40.38% | 99.21% | 0/40 supported |
| 1 | 10 ms | 64.00% | 98.36% | 88.94% | 40/40; median 11.976946 mm |
| 2 | 20 ms | 2.50% | 36.66% | 99.37% | 0/40 supported |
| 3 | 10 ms | 64.04% | 98.35% | 89.30% | 40/40; median 11.974901 mm |
| 4 | 20 ms | 2.00% | 37.21% | 99.33% | 0/40 supported |

All long-exposure refusals remain recorded. Short-exposure floor RMS medians
are 0.1213/0.1221 mm (worse than the first view's 0.0546 mm), with independent
plane-angle medians 1.111/1.013 degrees. Tiny separation spread is not precision
of every recovered point, much less absolute accuracy.

| Plateau support | Stage 1 separation min / median / max (mm) | Stage 3 separation min / median / max (mm) |
| --- | --- | --- |
| Primary | 11.960933 / 11.976946 / 11.985313 | 11.962519 / 11.974901 / 11.989734 |
| Near-upper | 12.027011 / 12.044971 / 12.057496 | 12.028007 / 12.052004 / 12.065724 |
| Left | 11.983608 / 11.993373 / 12.012628 | 11.977215 / 11.991462 / 12.004165 |

Every support measures all 80 short-exposure frames and reproduces primary
values exactly. None is selected for closeness to calipers. Between stages 1
and 3, common stable floor/plateau pixels have p90 absolute median-map changes
0.0425/0.0250 mm. Within-stage first/last common pixels have p90 differences
0.08/0.09 mm in the two short-exposure stages. These support a stable second
view during acquisition without fitting or subtracting motion.

## Corrected external-surface check

The initial external-height attempt incorrectly selected two turntable patches
below the visible object. It returned roughly 22.2--22.5 mm separation and is
**excluded as a measurement of the printed object's nominal 16 mm external step**.
The RGB context, reference layout and independent raw plateau-relative height
map resolve the semantic error. `external-height.json` retains every fit and
refusal; `SURFACE-IDENTITY-CORRECTION.md` records the correction. It also corrects
the historical `base` return-control label in both exposure blocks. A plane can
fit convincingly and still be the wrong physical surface.

The actual printed-base interiors are above/right of the plateau in this view:
primary **[285,80,325,110]**, left **[250,40,280,70]**. Their 1,200 and 900 pixels
return in every frame of every exposure stage. All six fixed base/plateau pairs
are evaluated on all 200 frames with freely fitted planes. Base is the reference
plane; separation is evaluated at the retained plateau centroid, with no nominal
height supplied to the fit. All 480 short-exposure pair fits have support.

| Base / plateau support | Stage 1 median height (mm) | Stage 3 median height (mm) |
| --- | ---: | ---: |
| Primary / primary | 15.822412 | 15.832850 |
| Primary / near-upper | 15.850915 | 15.862607 |
| Primary / left | 15.779013 | 15.792306 |
| Left / primary | 15.958085 | 15.963040 |
| Left / near-upper | 16.005586 | 16.003346 |
| Left / left | 15.955877 | 15.963048 |

Support dependence remains about 0.23 mm, despite good local plane residuals;
extrapolated separation is not an accuracy certificate. No owner caliper reading
exists for this particular height. Do not call its departure from nominal CAD
a measured print error or use it to choose a global scale correction. At 20 ms,
primary plateau RMS is 0.513--0.525 mm and primary base RMS 0.105--0.112 mm;
at 10 ms they are about 0.039/0.025 mm. Visibility and planarity improve together
on these observed printed surfaces. The current frame still crops the target,
so remaining faces and complete component dimensions are not qualified.

## Next bounded practical test

Reframe the **whole object in the depth view**, with the camera supported and
turntable stopped. No further pocket adjustment or new caliper reading is
required first. Verify a short fixed-view 20/10/20 exposure bracket at that
actual framing/range, identifying object and background separately. If short
exposure retains useful object coverage and geometry there, proceed to a
controlled full-turn comparison using fixed 20 ms frame period, gain 24 and
the already recorded X/Y dimensions. This is not permission to infer scene
motion or skip the stopped-view check from a vague "done".

Use fresh exclusive evidence outputs, actual delivery timestamps, the corrected
quiet lifecycle and first-fault guard, and the existing fixed-axis reconstruction
without fallback or fitted scaling. Compare dimensions, coverage, observed-plane
quality and repeated-angle closure; require successful final baseline restoration.
No merged exposures, firmware changes, experimental mesh or M27 resumption.
No global default is promoted before the full-view result and practical tolerance
are established. Another independent physical span remains conditional, not a
new prerequisite solely because the earlier external-surface label was wrong.

## Closure and prepared successor

At 20:45 UTC, the final audit verifies all 319 files from the earlier pending
checkpoint unchanged; both acquisition source snapshots, every recorded analysis
input/source identity and timestamp, and all 449 frozen live/saved context files
also verify. All three earlier sealed baseline/repeat/size-bias bundles retain
their exact file lists and hashes. M27 remains stopped. The scanner is present
as USB device 27 at the read-only audit; no new controls are sent for closure.

Both current-view and exposure evidence roots are now **SEALED together**,
20:46:46 UTC: **593 files / 237,584,657 bytes**. The manifest is
`/home/ben/scans/diagnostics/20260909-pocket-exposure-Fe9kV9/evidence-manifest.json`,
SHA-256 **`13e9db018ec45187127236fba1f480f9c39cf5618fbc2be3af5e07278777f3e0`**.
Its complete file list, sizes and hashes were independently rechecked after
creation. Integrity is not measurement correctness; the semantic correction
and earlier failed lifecycle remain part of the sealed record.

The successor is prepared, not hardware-executed:
`/home/ben/scans/diagnostics/20260909-component-exposure-fullview-50rD0l/`.
Its protocol and explicit-execution driver use the same quiet acquisition helper
for a stopped 20/10/20 ms bracket. Seven new synthetic preflight cases pass
(baseline, wrong timing/gain, thermal cutoff, wrong identity, unavailable device,
missing execution flag). These are separate from the earlier 59 software cases;
neither count is a physical trial count. Whole-target framing is required before
execution. Future data go in this fresh root, not either sealed root above.
