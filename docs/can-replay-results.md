# Historical can replay — 2026-09-10

Subsequent checkpoint: the [spatial-support/tail influence test](can-support-results.md)
is complete in a separate evidence root. Its ideal exact-pixel controls pass
for the 14 historically usable frames; one fixed real-tail refit does not solve
undersizing. This original recovery/replay record and its sealed inputs remain
unchanged; the next-test wording below is the decision at this earlier checkpoint.

## Outcome and acceptance status

The original August 24 measurement commands were recovered, including point
selection, stride and axial cropping. All **16 retained raw frames** reproduce
their original individual diameters to the recorded three decimals. Frozen
historical and current source trees produce **exactly identical** first/final
fit geometry, selected-point hashes and counts on every case. This is 32 offline
processes over 16 old inputs, not 32 captures or new independent measurements.
All processes complete in a combined 41.33 seconds, with no timeout/failure.

Every fit converges, but **all 16 remain INCONCLUSIVE under the original
validation rules**, both at their old 0.5 mm comparison threshold and the new
0.174 mm threshold (0.3% of the 58 mm can). Every point-estimate diameter is
outside the new threshold too; the two badly returning sprayed cases remain
unusable, not accepted diameter readings. A process exit, convergence, small
frame-to-frame spread and external dimensional qualification are different facts.
No [component acceptance criterion](component-acceptance.md) is newly passed.

The immediate consequence is specific: these single-frame discrepancies do not
require multi-view tracking and are not caused by later changes to the replayed
measurement code. This does NOT distinguish device depth/calibration, optical
surface effects, nonideal can geometry, coating or point-selection/fitting bias.
It does not prove that the entire current printed-component error has one cause.

Evidence root: `/home/ben/scans/diagnostics/20260910-can-replay-2JXPFA/`.
Sealed at 2026-09-10 14:25:51 UTC: **306 files / 15,248,270 bytes**, excluding
the manifest itself; manifest SHA-256
`d3d97cda195bdd9c73fc938047fcc7ad5fbfd1302698356884559ef5a216b553`.
Independent read-only verification confirms the exact file set, sizes and every
hash. Do not add or modify files in this root. Archived document snapshots
precede this seal metadata and the verification-order clarification below.
See `replay-audit.json`, `recovered-commands/`, `results/`, both source snapshots,
the preregistered protocol/addendum, execution logs and test reports. This root
contains no scanner acquisition. No firmware, calibration, preset, production
source or experimental mesh changed; M27 remains stopped.

## What was recovered

The original scanner-specific August 24 task record contains the actual commands
and corresponding returned output tables. Forty relevant command/result records
are retained with source line, timestamp, record identity and hashes. The full
task transcript is not copied and none of its historical shell actions is run.
In particular, archived capture, device, copy and commit commands are evidence,
not instructions for this replay.

Historical Git commit `31b7ee80f8ebc6aa67fed87cef4a3a2580386bad`, available in
another retained history, contains all 16 current raw payloads byte-for-byte.
That includes the rejected `can-sprayed-125-1.bin`, which is absent from the
15-frame manifest roster. The earlier September 9 review had preserved this
file but had not recovered its dated Git binding. The old scratchpad directory
is gone; its surviving preservation commands identify which files were copied.

The fitter, scalar depth conversion and cloud-export source files themselves
are unchanged from that commit. Other import dependencies evolved, so the
replay explicitly checks separate historical/current source snapshots rather
than assuming that identical fitter files imply a wholly identical environment.
Python version and all imported scanner-module source identities are retained.

## Exact measurement procedure

1. Decode all nonzero raw pixels in their original row-major order using the
   unchanged factory intrinsics and depth scale 0.004999999888241291.
2. Sort their Z values. Select values at `int(0.02*N)` and `int(0.98*N)` and keep
   all points inclusively between those values. These are exact order statistics,
   not interpolated percentiles or a depth band chosen from expected geometry.
3. Bare 98/120: fit all retained points, with no thinning or axial crop.
4. Bare 244 / sprayed 242: fit `band[::5]`; use that fit's centre and axis to
   crop the full band at strict `abs(axial) < 35 mm`; fit `cropped[::5]`.
   Preserve the uncropped first fit as well.
5. Sprayed 95/125: fit `band[::5]`, without the axial crop, retaining their
   historical unusable disposition and present inconclusive verdicts.

No fitting parameter, intrinsic, depth scale or selection limit is searched to
match an expected number. The copied commands establish the method independently
of the subsequent numerical matches. Camera point conversion still carries the
old `UNVALIDATED_UNIT` metadata label for faithful replay; the external caliper
comparison is in mm. Do not read the stale module prose as current project status.

## Reproduced retained-set measurements

Ground truth remains the owner's bare can body reading **58.0 mm**, five readings,
recorded uncertainty **0.05 mm**. Standoff labels below are scanner-derived,
not independent physical distance measurements. Coating thickness was not
independently measured; do not assume the sprayed surface is certified 58.0 mm.

| Retained subset | Frames | Mean diameter (mm) | Mean error (mm) | Relative error | Range across frames (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bare ~98, hub-era | 2 | 56.640545 | -1.359455 | -2.3439% | 0.017893 |
| Bare ~98, direct-port retained subset | 3 | 56.663976 | -1.336024 | -2.3035% | 0.014697 |
| Bare ~123 | 3 | 57.704697 | -0.295303 | -0.5091% | 0.043920 |
| Bare ~244, cropped | 3 | 57.441476 | -0.558524 | -0.9630% | 0.011064 |
| Sprayed ~242, cropped | 3 | 57.716585 | -0.283415 | -0.4886% | 0.017814 |

These are computed means of the explicitly retained subsets, not replacements
for differently sized historical groups. All underlying per-frame results and
original rounded output lines are retained. The single retained sprayed-95 fit
reproduces 29.374 mm; the single retained sprayed-125 fit reproduces 61.934 mm.
Both remain unusable as measurements of the 58 mm body.

The historical validation uses twice the radial RMS as its fit-uncertainty
heuristic, plus checks for structured residual, conditioning and convergence.
That heuristic is preserved, not promoted into a statistically calibrated
uncertainty budget. The best sprayed cases still trip the structured-residual
check. The 0.3% criterion is not softened to make these inconclusive fits pass.

## Corrections to reading the old record

- The 56.689 mm direct-port mean used seven frames; only three of those survive.
  Near fixture files 1/2 are the earlier hub pair, not two of those seven.
- Both far means used four frames, but only frames 1–3 were retained. In
  particular the bare-far manifest associates a four-frame mean/spread with
  `n=3` and three listed diameters. The original output resolves the discrepancy.
- The sprayed-125 headline 62.01 mm is a four-frame mean. The recorded 2.82 mm
  fit uncertainty belongs to the first frame, whose diameter is 61.934 mm.
  They must not be reported as one fit.
- The recovered ideal-cylinder control used the correct factory intrinsics,
  but a clean synthetic surface at X=0, without the real spatial dropout/noise
  pattern. It checks ideal-case arithmetic. It does not eliminate estimator
  bias on noisy, uneven or contaminated real returns. Its sweeping claim that
  the remaining error therefore belongs to the sensor is not established.
- The earlier principal-point and calibration-sweet-spot causal claims remain
  withdrawn/weakened. A surface-dependent residual sign change alone does not
  uniquely identify the mechanism, and changing range did not independently
  control surface response, image support or physical standoff.

These clarifications preserve the original records instead of editing their
historical numbers into apparent agreement.

## Next decision, bounded

Do not spend another acquisition merely proving that a can can measure short.
That fact is now reproducible. The next selected question is whether the
single-frame measurement is materially biased by the spatial support and
outlying returns it fits, before proposing a calibration correction.

Start from these saved frames and frozen procedure. A useful next control would
pair a known-geometry synthetic cylinder with the observed pixel-support masks,
and report residual/support structure on the real points without fitting scale
to the calipers. Predeclare the cases and point-selection rule. Passing an ideal
or mask-only control would rule out only that mechanism, not real optical/noise
bias; an inconclusive result must name the additional physical reference it
needs. This is the next diagnostic decision, **not a test executed in this root**.
There is no present justification for a firmware flash or a global scale tweak.

## Verification

Nine wrapper/input/synthetic tests passed before the real replay. Four further
independent-output-parser/denominator tests bring the combined total to **13**,
all passing; the repeated nine are not additional independent tests. The audit
checks all original rounded per-frame diameters, all available uncropped rounded
diameters, exact historical/current geometry and selection parity, all imported
module hashes, 175 archived input/source/command records and 78 current source
files unchanged. It retains all 32 process logs and their hashes. The scanner
was not activated, so no new scanner-temperature measurement is claimed.

Closeout also verifies all 449 frozen live/saved M27-context files and observes
the owned M27 process still stopped. The first final inventory assertion used
different ordering conventions for Path objects and relative-path strings;
`execution/` versus `execution-plan.json` exposed the mismatch after the manifest
was written. No member or hash differed. An independent read-only check sorts
both relative-name lists consistently, verifies unique membership and all 306
sizes/hashes, and passes without modifying the sealed root. This was a verifier
ordering defect, not a replay/input failure; the original closeout source and
manifest remain intact. Repository whitespace checks pass.
