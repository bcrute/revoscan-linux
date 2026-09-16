# Whole-object exposure check — 2026-09-09

Subsequent outcome: the [full-turn comparison](component-turn-exposure.md) now
runs the previously prepared bracket. Acquisition is clean, but the ordinary
10 ms reconstruction is substantially undersized; its four-round estimator
returns before meeting its own convergence threshold. A separately recorded
continuation reaches that threshold at round six; a shared-fixture control
then restores short-exposure widths to the baseline average but leaves the
common undersizing. The
stopped findings below remain local-quality results, not a promoted preset.

## Current outcome

The fresh stopped retry **does support a local surface-quality improvement at
10 ms exposure**, with frame period fixed at 20 ms and gain 24. Four of five
selected object patches support planar fits; their full-input residuals improve
by approximately 1.6–4.6 times, without point loss on those four patches. The
fifth patch refuses every plane fit and remains reported, not replaced by a
more favorable patch. This is not yet a full-object dimensional qualification.

The parent evidence root below includes the successful retry under
`stopped-retry-1tOcGP/`. Its `QUALITY-PROTOCOL.md`, annotated regions, 120 raw
frames, per-frame fits/refusals, common-support comparisons, stability checks
and return-loss map preserve the method and limitations. The earlier rotating
attempt remains unchanged and refused.

The parent bundle is now sealed (2026-09-10 00:16:40 UTC): **485 files /
141,995,921 bytes**, manifest SHA-256
`f5948fd3345e99938c20aeca1d5bc315e360285443f2288b83f482cfd1060616`.
Exact file list, sizes and hashes verify. Both captures' source/snapshot
identities, the quality inputs, four earlier seals and all 449 frozen live/
saved context files also verify unchanged. At this seal checkpoint the next
turn-test root was prepared and unexecuted; it has now run separately and is
not part of this seal. No files may be added to or
rewritten inside the sealed full-view root.

## Genuinely stopped retry, 23:51–23:52 UTC

At 23:51:44 UTC the exact MINI and original 20/20 ms/manual mode 0/gain 24
verified. The same 20/10/20 ms bracket retained 40 settled frames per stage,
after at least one second and 12 discarded frames at every transition.

- 160 full deliveries; 120 settled raw frames; zero drops or short frames.
- Observed cadence 4.524 fps. This does not increase the frame rate.
- Maximum CPU/NPU including final readback: 60.2/59.5 C. Projector temperature
  is not measured; the test's 80 C cutoff is not a manufacturer rating.
- Quiet shutdown, original timing/gain restoration, post-close identity and
  source-byte/time checks passed; no transport fault.
- First versus every later same-stage frame has maximum p90 absolute depth
  change 0.115/0.090/0.115 mm. First/last views also retain the same object pose.
  This supports the stopped condition, not independent pose/accuracy truth.

Median per-frame RMS of **all selected valid points about each independently
fitted plane**, in millimetres:

| Fixed image region | Initial 20 ms | 10 ms | Final 20 ms |
| --- | ---: | ---: | ---: |
| Large triangular face | 0.0553 | 0.0186 | 0.0546 |
| Same face, left support | 0.0373 | 0.0240 | 0.0373 |
| Near block, right face | 0.0272 | 0.0132 | 0.0275 |
| Near block, left face | 0.0621 | 0.0134 | 0.0614 |

All 120 fits pass for each of these four patches. Every selected pixel returns
in all initial-20 and 10 ms frames. Only final-20 block-left loses a few
intermittent points (99.978% mean returns). Fixed shared-support fits reproduce
the same improvement, and all-input residuals expose points excluded by robust
fitting. The selected lower-plate patch returns 100% at every exposure but all
120 fits refuse as **not convincingly planar**. Do not infer its cause or choose
a replacement after seeing the result.

The candidate also shifts measured depth: these four patches have candidate-
versus-initial median-map absolute-change p90 of 0.075–0.1175 mm, compared with
0.030–0.050 mm for final-versus-initial 20 ms. Lower residual is not proof that
the shifted surface is absolutely correct. No temporal median enters a plane
fit; median maps are a separate raw same-pixel comparison.

## Coverage caveat and next test

The explicit turntable window drops from 64.0% returns to 1.0%, then recovers
to 64.1%. A mixed between-feature window changes from 32.45% to 31.56% to
32.55%; it is not an exact semantic surface-coverage measurement.

Across the whole image, 78,358 pixels return at least 90% in BOTH 20 ms
brackets. Of those, 3,805 return less than 10% at the short exposure. The
`whole-view-return-changes.png` overlay shows substantial losses around the
turntable, **but also along the object's perimeter and near the ramp edges**.
Those pixels are not independently labeled true surfaces versus mixed/spurious
edge returns. Do not call all losses background or claim no meaningful depth
loss globally. Three persistently gained pixels and 84 stable-overlap pixels
with greater than 0.5 mm median-depth change are also preserved; the latter
are changes, not calibrated errors.

The foreground lower corner still reaches the image bottom. A little more
lower margin is needed before qualifying full rotations. The next operator
action is to aim/recenter so the object moves slightly upward in the depth
image, retaining room above the ramp, and start the turntable slowly with the
scanner supported. Pocket-floor visibility is not required. No new caliper or
replug is needed.

A separate prepared root contains the conditional full-turn comparison:
`/home/ben/scans/diagnostics/20260909-component-exposure-turn-3mDF8j/`.
Its protocol fixes 20/10/20 ms exposure, 810 settled frames per stage in one
stream, a 2600-delivery/600-second ceiling, original gain/frame period, thermal
monitoring and guarded restoration. It was unexecuted at this checkpoint;
the subsequent hardware run is linked above. Eighteen
capture/preflight synthetic cases and six quality-reporting cases pass (24
total, not hardware trials); a first test run's four old-filename assertions
and the corrected run are both retained. The short-run seven preflight cases
also passed, but their rerun is not seven additional distinct tests.

Each future exposure stage is reconstructed independently using the same
predeclared 405 even-indexed frames, actual delivery times, fixed-axis method
and unchanged dimension/surface checks. Compare against both long-exposure
brackets and existing platform X/Y calipers 91.77/81.74 mm. Evaluate coverage,
edges and dimensions as well as smoothness. No global setting, calibration,
production change, firmware, merged exposure cloud, experimental mesh or M27
resumption is promoted by the stopped result.

## First attempt, 21:10 UTC — acquisition succeeded, comparison refused

The first whole-view attempt acquired cleanly but **cannot establish an exposure
improvement**: the target rotated throughout the intended stopped comparison.
No dimensional or per-region quality result is accepted from these frames.
This is a setup-condition failure, not a USB/thermal failure or evidence that
the earlier pocket improvement disappeared.

Parent evidence root (contains both attempts):
`/home/ben/scans/diagnostics/20260909-component-exposure-fullview-50rD0l/`.
`stopped-first-last.png` shows progressive rotation within all three stages;
`motion-refusal.json` separates acquisition success from experimental refusal.

The operator's clarified goal was the whole-object setup, without requiring
visibility of the pocket floor. The initial GUI preview shows the whole object
at that orientation. The operator closed the GUI before the agent's planned
window-close request; that request exited before acting on a now-absent PID.
The agent then observed the known quiet boundary before any control reads.

At 21:10:23 UTC, exact serial A22704522A6F00A64, original 20/20 ms/manual mode 0,
gain 24 and CPU/NPU 60.7/60.4 C verified. The fixed 20 ms frame-period sequence
uses exposures 20/10/20 ms, 40 settled full-resolution frames per stage, with
12 discarded frames and at least one second after every transition. Results:

- 160 full deliveries; zero sequence drops or short frames; 120 settled raw frames.
- Observed whole-block cadence 4.525 fps; no claim of 10 fps.
- CPU/NPU maximum 61.6/60.8 C; projector temperature is not measured.
- Every transition's timing/gain and stage-ending gain verified.
- Quiet shutdown, fresh final baseline/gain readback and post-close identity passed.
- No transport fault, firmware/default/calibration/production change or mesh.

The capture report's stationary-confirmation flag is the intended operator
condition, not independent verification. Actual raw frames contradict it.
Because both viewpoint and exposure changed, no fixed-ROI return/plane comparison
is run and no motion correction is fitted to manufacture a stationary trial.
Actual delivery times and all raw frames remain usable historical evidence.

The rotating views also bring a lower base corner to the image's bottom edge.
Whole-object framing at one orientation does not yet establish room for a turn.
The next action at that checkpoint was to stop the table and recenter to leave
more lower margin, keeping a supported scanner and unchanged scene afterward.
No new caliper reading, replug or special pocket angle is required. Repeat the
same bounded stopped bracket to fresh outputs before advancing to a full-turn
comparison against the existing physical X/Y dimensions. M27 remains suspended.
The stopped retry above now completes that comparison; lower framing margin
and the actual full-turn comparison remain pending.
