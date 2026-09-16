# Full-turn exposure comparison — 2026-09-10 UTC

## Primary outcome

The long 20/10/20 ms bracket acquired cleanly. **The ordinary four-round 10 ms
reconstruction is not acceptable**, despite the earlier improvement on stopped
raw planar patches. It is approximately 3 mm undersized and has worse global
shape/orthogonality. A specific software limit is implicated: the estimator
returns after four rounds while its rotation-centre update is still 6.278 mm,
well above its own 0.5 mm convergence criterion. Both 20 ms fits settle below
0.2 mm. A separately recorded continuation of the SAME saved candidate model
converges at round six and repairs much of the distortion, but remains about
1.0–1.1 mm undersized. A further, explicitly combined fixed-fixture control
brings short-exposure widths within 0.003 mm of the two 20 ms widths' average,
while retaining roughly 0.63/0.75 mm shortfalls. This localizes the candidate's
EXTRA size error to its fixture estimate in this test, without identifying the
shared residual bias as hardware, scale or a uniquely known mechanism.
No preset, scale correction or general scanner qualification is promoted.

Evidence root, SEALED:
`/home/ben/scans/diagnostics/20260909-component-exposure-turn-3mDF8j/`.
The predeclared acquisition/evaluation protocol, all sources, actual sample
times, raw frames, wire trace, stage fits, failures and continuation are retained.
No firmware, calibration, global default, experimental mesh, production-source
change or M27 resumption has occurred.

Seal: 2026-09-10 03:06:33 UTC, **4,742 files / 2,503,260,117 bytes**;
manifest SHA-256
`8446033ecd1d95c763302592400a5e865aff59e543811ca6fcd189991e0e4033`.
Exact file list/sizes/hashes verify. All 2,430 raw payloads, 14 live mirror
payloads, 91 capture source/snapshot records, five cases' input/cache/selection
and proper-rigid-pose checks, all five earlier seals and all 449 frozen live/
saved-context files verify unchanged. Closeout observes USB device 34 at
480 Mb/s, GUI closed, no capture process, M27 stopped. No control read was
issued for that final filesystem/sysfs check. No further files may be added
inside this sealed root; any next work needs a fresh evidence root.

## Acquisition and framing

Owner confirmed the added lower margin, slow running turntable, supported
scanner and closed app. Initial 02:22:47 UTC: exact MINI A22704522A6F00A64,
original frame/exposure 20/20 ms, manual mode 0, gain 24; CPU/NPU 58.9/58.2 C.
The prepared helper is unchanged. A separately tested outer adapter mirrors
occasional raw deliveries for live framing inspection, after the original
callback. No tracking/rendering is added to the capture process. Those rare
512 KB writes mean this is not a pure maximum-throughput benchmark.

- 2,470 complete deliveries, zero sequence drops/shorts, 2,430 distinct settled
  payloads (810 per exposure), totaling 1,244,160,000 raw bytes.
- Fixed 20 ms frame period and gain 24, exposure 20/10/20 ms, original settings
  restored; quiet stop/fresh readback/post-close identity passed, no transport fault.
- Whole-block observed cadence 4.544 fps. Settled stage rates are 4.5205 /
  4.5994 / 4.5206 fps, not 10 fps or a general throughput fix.
- Peak and final CPU/NPU 63.8/63.0 C. The last-minute linear slopes are still
  approximately +0.61/+0.30 C per minute: no thermal equilibrium claim. The
  80 C cutoff is a test policy, not a manufacturer rating, and projector
  temperature is not measured.
- All 91 capture source identities and saved byte/timing checks pass. Raw
  frames were written after capture except explicitly labeled live mirrors;
  neither file-write time is substituted for actual callback/delivery time.

Sampled angles show the lower printed base inside the frame. The upper ramp
has very little margin and approaches rows 2–4 for some angles. A raw row-zero
test alone misses this concern: every frame has some unsegmented row-zero
returns, but none an eight-pixel run. The separate top-extent check finds a
broad in-band return at or above row 4 in 172/118/106 frames of the three
stages. These are clipping warnings, not independent semantic labels. Do not
claim generous optical margin, full-height/upper-edge qualification, or that
bottom-row turntable returns are a clipped printed base. Primary footprint
measurements and their broad sidewall support are retained separately.

## Independent primary reconstructions

Each stage uses its predeclared 405 even-indexed settled frames and the
unchanged 60–300 mm preparation, fixed-axis algorithm and saved-cloud scorer.
No cross-exposure poses, point mixing, scale fitting or fallback are used.
The scorer repeats its earlier full/reduced-SVD equivalence check. Historical
aggregate-gate failures are retained; completed dimension checks are not
suppressed merely because that old surface threshold fails.

| Quantity | Initial 20 ms | 10 ms, original four rounds | Final 20 ms |
| --- | ---: | ---: | ---: |
| X span, mm (caliper 91.77) | 91.10698 | 88.74710 | 90.93869 |
| Y span, mm (caliper 81.74) | 81.19642 | 78.54267 | 81.01678 |
| X error, mm | -0.66302 | -3.02290 | -0.83131 |
| Y error, mm | -0.54358 | -3.19733 | -0.72322 |
| CAD-surface RMS, mm | 0.45277 | 1.15424 | 0.46936 |
| CAD-surface p95, mm | 0.69952 | 2.41522 | 0.73960 |
| Observed reference fraction | 95.325% | 94.925% | 95.475% |
| Surfaces passing / judged | 23/24 | 19/24 | 23/24 |
| Worst orthogonality deviation | 1.042 deg | 6.341 deg | 1.216 deg |
| Pocket-floor observed fraction | 7.27% | 0% | 12.73% |
| Final centre step, mm | 0.12003 | 6.27846 | 0.18586 |

X/Y above use the fixed 3 mm face-selection half-window at platform mid-height.
The 5 mm control reproduces both 20 ms widths exactly; the failed candidate's
Y shifts another -0.0309 mm at 5 mm. This does not rescue it. The two baseline
widths differ by 0.1683/0.1796 mm; a claimed candidate improvement must be
judged against that disagreement, not compared only to the worse bracket.
Calipers remain owner-reported, without a formal uncertainty budget. The CAD
retains its pre-existing physical-print scale correction exactly once.

All three models cover more than one turn (approximately 1.59/1.56/1.59).
Nearby-frame residuals are 0.0778/0.0480/0.0824 mm, while approximately
60-degree-pair agreement is 0.2204/0.5878/0.2341 mm. Good nearby and full-loop
agreement did not reveal the candidate's global geometric distortion.

Timing clarification: actual stage timestamps are retained, supplied and
round-trip through the saved project exactly. The unchanged frame_angles helper
snaps to frame indices if cumulative normalized deviations remain below 0.1
interval. All three stages take that branch, hence the model's "frame order"
label. Do not claim unsnapped timestamps drove these poses. The primary
rotations_from_actual_times field is an approximate span from supplied times;
the actual snapped model-angle span is separately checked during continuation.

## Targeted continuation (secondary, not the primary comparison)

CONTINUATION-PROTOCOL.md fixes one test after observing the primary failure:
restore the four-round 10 ms model exactly, then allow at most four more
unchanged pair/rate rounds, using the same 0.5 mm stopping threshold. All 405
original poses and all 834,913 original registered points reproduce exactly
before any further estimation. No new scanner access or point/CAD changes.

Round 5 moves the centre 1.1074 mm; round 6 moves it 0.1929 mm and stops.
The approximately 60-degree agreement improves from 0.5878 to 0.2083 mm;
approximately 90-degree agreement remains worse at 0.7495 mm. This is a
bounded convergence improvement, not yet proof of acceptable dimensions or
complete surfaces. The same dimension/surface evaluation completes to fresh
continued-10ms/stage-1 outputs: X/Y 90.67045/80.74270 mm, errors
-1.09955/-0.99730 mm; CAD RMS/p95 0.48206/0.70172 mm; 24/24 judged surfaces
pass the old surface threshold, worst orthogonality 0.76175 degrees, observed
reference fraction 95.125%, pocket-floor coverage still zero. The 3/5 mm
dimension windows agree exactly. Both primary failure and continuation remain
available. Convergence improves the result, but does not make the short-exposure
dimensions better than either long-exposure baseline.

## Explicitly combined fixture control

FIXTURE-TRANSFER-PROTOCOL.md fixes one secondary control, not a retrospective
replacement for the independent primary bracket. Use equal weights for BOTH
converged 20 ms fixture lines: align unit-axis signs, average/normalize the
axes, average their closest points to the camera origin and project that point
perpendicular to the mean axis. The source axes differ by 0.01802 degrees and
their closest points by 0.17452 mm. No baseline is selected by its caliper
agreement; no CAD, caliper or scale enters the fixture calculation.

Only the original 405 short-exposure frames supply points. Their continued
short-exposure rate and unchanged timing interpretation supply relative angles;
the first short frame stays the identity pose. No long-exposure point is added.
This shares a mechanical-fixture estimate and is NOT an independent short-
exposure fit, a general handheld solution, or an independently measured axis.

Results in fixture-transfer-10ms/stage-1:

- X/Y 91.02004/81.10653 mm, errors -0.74996/-0.63347 mm.
- Difference from the two baseline widths' mean: -0.002793/-0.0000687 mm.
- CAD RMS/p95 0.44503/0.64547 mm, 24/24 judged surfaces passing, worst
  orthogonality 0.87353 degrees; observed reference fraction 95.0833%.
- Pocket-floor coverage remains zero; do not treat the aggregate surface
  threshold as whole-target completeness or dimensional qualification.

The combination improves some shape measures and removes most of the
short-exposure fit's additional size error. It does NOT beat the better
20 ms width result beyond the baseline disagreement or solve the common
undersizing. No further weight/axis/scale sweep follows this control.

An equal-scale plot of the measured mid-height base points is saved as
base-footprint-comparison.png. Its dashed caliper rectangle is a display guide,
not fitted geometry; fixed display/height bounds and every plotted source hash
are recorded. The initial distorted short reconstruction is shown as well as
the corrections, not silently discarded.

## Decision and remaining dependency

Keep original 20/20 ms, manual mode 0, gain 24 as the hardware baseline. Do not
promote universal 10 ms or the shared-fixture workflow from this one setup.
The software lesson is concrete: a capped fit must not be called converged
because it returned a model; the diagnostic continuation demonstrates the
effect of two additional unchanged rounds. These were offline diagnostics,
not an installed production fix. Likewise, all scorer processes return zero
on completed evaluation even when JSON accuracy/orthogonality verdicts fail;
parse those verdicts rather than treating process exit zero as qualification.

The owner's required dimensional tolerance is still unknown. It was requested
asynchronously after seeing the remaining approximately 0.6–0.8 mm bias, so the
next decision can be practical component acceptance versus tighter metrology,
not another open-ended sweep. No tolerance is invented. Before any further
whole-object qualification capture, increase view margin (roughly 2 cm farther
away with similar aim, rather than repeated tilting) to keep both base and
ramp comfortably in frame through a turn. That action has NOT yet been requested
as a new completed setup or performed. The last confirmed table state remains
running; no further camera acquisition occurred during the offline tests.

Twenty-one acquisition/monitor tests pass (18 reused prepared cases plus three
mirror tests), twelve offline selection/dimension checks pass, and three
fixture-mean tests pass. These are software cases, not hardware trial counts.
No experimental mesh, firmware flash, calibration/default/production-source
change, owner-file overwrite, commit/push or M27 resumption occurred.
