# Operator review checkpoint — 2026-09-09 16:18 UTC

The operator requested a pause to discuss progress versus churn and whether the
scanner hardware is responsible. No additional experiment is authorized by this
discussion. Detailed evidence is preserved; no release-qualified fix exists.

## Process and source state

The M27 controller, PID 2009261 / session 32082, was explicitly command-validated
and suspended with SIGSTOP at approximately 16:16:36 UTC. It cannot launch more
queued cells while suspended. The two then-active children, 2254313 and 2257375,
were allowed to finish their existing recorded replays; both are exited/zombie
at the 16:18 observation. Do not mistake a suspended controller for completed
matrix execution. No active scanner acquisition or firmware operation occurred.

Machine-readable inventory and exact process/source/output identities:
`/home/ben/scans/diagnostics/20260909-reliability-organized-model-stage-b/discussion-checkpoint.json`.
The one-shot `checkpoint-discussion.py` produced it successfully at 16:18:34.
Both frozen source sets still match all live and saved bytes: M27's 447 context
files and the separate historical wrapper's 449-file union. Neither root is sealed.

M27 has all twenty observation controls and ten separately acquired original
parents exact. Its eighty changed-run matrix has 37 saved per-cell score files
at pause; no final matrix execution/audit exists. Late child reports are retained
but have not passed the suspended parent's scoring/validation. Early favorable
combinations are not independently audited whole survivors or Stage B candidates.
The fixed-sample and interaction auditors are prepared and remain unexecuted for
the changed matrix. Completed M1–M26 aggregate remains 695 changed trajectories,
139 whole configurations and zero quality-qualified candidates. Do not add an
incomplete M27 phase to those totals.

The separate Stage B wrapper passed 113 focused and 2,006 selected tests, clean
lint and cold typing of 139 files. Those are software checks, not 2,006 scanner
experiments or accuracy results. There are no recorded M27 Stage B trajectories.
`campaign_common.py` is an INCOMPLETE draft; model-query validation/return and
control comparison are unfinished. The planned driver and auditor do not exist.
Do not execute or treat this infrastructure as ready because the wrapper passed.

## Interrupted input inventory and a concrete adapter limitation

The CPU inventory, PID 2219353 / session 30721, was intentionally stopped with
SIGINT for this discussion. It retained its partial JSON and exited 130. Its
`failed` / KeyboardInterrupt status is an operator-pause outcome, not a scanner
failure. It verified 7,011 raw input occurrences and started 14,022 preparations:
13,738 compatible, 283 incompatible and one interrupted/in-progress record.
The last 1,322 original inputs were not inventoried. Full readiness is not claimed.

All 283 incompatibilities are dense preparations above the nominal 1,800-point
target: development 16, old-adverse 249, partial later-adverse 18. Counts reach
1,834; retained clouds remain within their original budget. The original grid
sampler deliberately approximates its target using floor-derived stride/count,
so target is not a strict upper bound. The new FrameBinding.prepare and inventory
wrongly treat it as one. The inventory fails this assertion before checking
pixel identity on those oversized clouds. This is a diagnostic-adapter contract
mismatch, not evidence of bad hardware, nor an explanation of earlier tracking
failures that predate the adapter. No fix or gate waiver has been made.

## Assessment for discussion

Real gains include measured GPU/GUI throughput, reproducible acquisition timing,
identifying useful-depth losses at aggressive timing, and exposing false
acceptance: an uninterrupted scan can still have wrong geometry. Strong local
software evidence comes from the stationary face-A capture: the same saved raw
data gives approximately 27.39 nominal mm apparent movement with production
sampling versus 0.42 with hierarchical-grid sampling. That is not a hardware
replacement or a physical ground-truth calibration; it demonstrates substantial
sensitivity in our tracking pipeline. Face B and moving results show that sampling
alone is insufficient. See `reliability-view-b.md` and `live-gpu-throughput.md`.

Recent broad sweeps have diminishing returns. One earlier five-case survivor
failed all four longer geometry cases. More variations of the same local fitting
family cannot by themselves determine whether the measurement, scene assumptions,
or tracking architecture is the primary limitation. Reused development scores
and software test counts must not substitute for that causal distinction.

Hardware failure is not established. About 13 raw fps has already crossed the
same USB2 link, and severe pose errors also occur on stable saved data. Neither
fact rules out optical calibration problems or motion-dependent depth distortion.
There is no adequate matched Revo Scan baseline or independently known geometry/
motion for the current target in the closed evidence used here.

Proposed next decision, NOT executed: prioritize a small same-device comparison
against compatible official Revo Scan, with scanner fixed / turntable stopped
first, then one controlled motion, using a clearly identifiable, feature-rich
target and appropriate marker control. Compare an independent physical dimension
and each frame's geometry before interpreting accumulated shape. If official
software succeeds and ours fails, prioritize the host tracking architecture. If
both fail, separate scene/setup from calibration/optics using the stationary
measurement and a second target. Firmware flashing is not the current first step.

Manufacturer context, consulted for this discussion: MINI guidance distinguishes
feature and marker tracking; official object-handling guidance calls for markers
on symmetrical/featureless objects. This does not establish that our exact printed
object is featureless or reveal Revo Scan's internal registration algorithm.
[MINI tutorials](https://download.revopoint3d.com/support/download/mini/mini-tutorials-and-tips-en-v1.0.pdf),
[object handling](https://revopedia.revopoint3d.com/en/scanningtips/objecthandling).

## Resume only after direction

Revalidate PID 2009261, exact command and frozen source identities before any
SIGCONT. Its queued work is still the original eighty-cell plan, not a new plan;
resumed elapsed timings include suspension and are not FPS. If changing frozen
source or abandoning this matrix, explicitly close out the partial campaign
first and retain all reports. Do not blindly restart the exclusive-output driver.
Do not overwrite the partial input inventory; any corrected contract/inventory
requires an explicitly separate version and truthful before/after record.

## Subsequent operator clarification: mapper research value

After the pause, the operator clarified that the findings are also intended for
the mapper project and that generalizable research can justify work even without
an immediate scanner fix. The [transfer synthesis](reliability-mapper-transfer.md)
records the relevant evidence, proposed design implications and limits. This
changes the assessment of research value, not the outcomes of completed tests.
No batch resumption, new experiment or mapper implementation is performed during
this discussion; the process checkpoint above remains the last observed state.
