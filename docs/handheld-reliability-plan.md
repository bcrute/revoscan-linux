# Handheld depth-dropout investigation — 8 September 2026

## Objective and evidence

Make handheld tracking reliable without hiding missing depth, accepting more
drift, relaxing alignment gates, or introducing the abandoned live mesh.
This is a staged investigation, not a promise that a software change can recover
depth the sensor did not measure.

The move–pause recording `20260908-045034` contains usable quieter intervals but
ends with three alignment refusals (185 accepted / 188 offered). The preceding
combined handheld/turntable recording `20260907-181004` has 284 accepted / 303
offered and substantial raw-depth loss. Neither has a credible automatic revisit
test, so neither is a geometric quality pass. Camera pose, range, angle and
framing vary together; the recordings do not isolate physical motion sensitivity.

## Predeclared execution order

1. **Freeze evidence and reproduce the failure.** Keep the original captures and
   sealed reviews unchanged. Save the current replay/tracking/preparation/health
   source, plan, source hashes and results in a new diagnostic directory. Replay
   all 188 frames with the existing default and verify exact decisions and poses
   against the saved GPU baseline, excluding timing/report-only fields.
2. **Audit reference admission.** For each offered frame, compute raw nonzero
   yield, in-band yield, and the existing causal 15-frame return-health verdict.
   Record which accepted frames become keyframes and appear in actual candidate
   references. Distinguish accepted-frame indices from raw-source indices.
   Establish whether warning frames actually entered references before loss;
   temporal association alone is not proof of cause.
3. **One closed, offline-only reference experiment.** Use the existing health
   thresholds unchanged: no returns, less than 5% raw yield, or the existing
   relative-collapse/hysteresis rule. Prevent warning frames from becoming
   archival keyframes or contributing to multi-frame model/recovery references.
   Still offer every frame, retain every accepted pose, and preserve the
   immediately previous cloud and velocity semantics for adjacent tracking.
   Preserve the initial anchor even if it warns, explicitly recording that
   exception. No future pause labels, fitted motion priors, interpolation,
   acceptance-threshold changes, larger reference budgets or parameter sweep.
   This tests archival contamination, not every possible dropout strategy.
4. **Evaluate, then stop or validate.** Compare default and experimental replay
   on the 188-frame development recording, then the 303-frame contrasting
   recording. Both have already been inspected: neither is a blind holdout.
   If the experiment regresses terminal tracking loss on development, reject it
   for production; run the contrasting recording to document generalization,
   but do not spend time tuning around the failure. If it improves development
   without a contrasting regression, additionally run the 628-frame 20 ms
   turntable control and the 2,703-frame adverse moving recording against their
   existing default results and the unchanged 25-pair geometric audit. Earlier
   controls have also been used previously and must not be called untouched.
5. **Verification and promotion gate.** Unit-test unchanged default decisions,
   no-warning parity, causal classification, sparse-anchor handling, refused
   frame indexing, candidate independence, cloud/keyframe memory bounds, and
   explicit unqualified status for failed source recordings/diagnostic modes.
   Require no new loss or geometric regression on controls, and an independently
   checked physical return-to-start scan before any live tracking-policy change.
   More accepted frames, an inconclusive revisit, or a new software FPS number
   alone is insufficient. A failed experiment remains available as diagnostic
   evidence, never silently becomes the GUI default.
6. **Controlled physical sensor comparison — operator setup required.** Once
   the operator confirms the camera is fixed, the entire object/base stays in
   view, and the turntable can repeat one unchanged speed, compare 20/20 ms and
   14/14 ms timings at the same gain 24, auto exposure off. First use stopped
   table reference clips, then matched moving clips in A–B–B–A order. Start with
   60-second clips, at least 12 complete frames/one second settling, record
   timing/gain readback, every raw frame/hash/timestamp, valid-depth distribution,
   spatial dropout, temperature and USB continuity. Compare matched view bins,
   not just whole-run yield or delivered FPS. A handheld repeat must subsequently
   return to an identifiable starting pose; do not ask the operator to hold
   perfectly still. Physical displacement per second remains unknown without
   an external motion measurement.
7. **Thermal extension and escalation.** Only extend a meaningful-depth winner
   to 15 then 30 minutes after short comparisons pass; monitor the available
   CPU/NPU sensors every 5 seconds, flag 75 C and stop at 80 C (not a certified
   optics/projector limit). Restore original settings with fresh exact-serial
   descriptors and verified quiet lifecycle boundaries. Stop on the first
   transport fault; no automatic reset loop. USB 2 data negotiation and firmware
   frame limits are not themselves explanations for pose drift. Firmware
   flashing remains blocked until an authenticated image, calibration backup,
   verified recovery procedure and explicit concrete flash plan exist.

## Scope of this execution

Execute stages 1–5 offline now, respecting the early-rejection rule. Stage 6
requires a newly confirmed physical setup; do not infer that a previous
turntable/camera position remains unchanged. Do not stream or change device
settings while another application may own the scanner. No firmware writes,
experimental live mesh, commit, push, installation or persistent system change.
Report completed, rejected and waiting stages separately, with all failures
preserved. Record the outcome in a companion results document.
