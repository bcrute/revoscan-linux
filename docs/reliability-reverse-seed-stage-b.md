# M21: longer and archived handheld qualification

Continuation of [M21's complete Stage A result](reliability-reverse-seed.md).
Evidence root: `/home/ben/scans/diagnostics/20260909-reliability-reverse-seed-stage-b/`.
Complete, independently rescored at 2026-09-09 04:49:32 UTC: four exact routing
controls and all six changed replays finish. **The M21 Stage A survivor is
rejected in Stage B.** All 480 fixed scores reproduce exactly; every original
input is accounted for. Nothing is promoted to production.

Only the actual whole Stage A survivor advances: ordinary-900 / centered /
reverse-forward / grid-pooled-residual, model budget 1,800, ten-neighbor GPU
normals, original gates and fifty iterations per direction. M22–M24 remain
deferred. The predeclared protocol, exclusive plan and source snapshots bind
the settings before these runs; no per-recording tuning or retries with relaxed
limits. At most two GPU children execute concurrently.

## Frozen evidence and guard

| Original recording | Required original inputs | Fixed raw-pair records |
| --- | ---: | ---: |
| Slow turntable D | 1,609 | 65 |
| Development extended | 1,171 | 103 |
| Old adverse | 2,703 | 165 |
| Later adverse | 2,359 | 147 |
| Handheld with pauses | 188 | 0 |
| Combined handheld motion | 303 | 0 |
| Total | 8,333 | 480 |

The original source catalog, input verification and 480-pair reproduction are
copied byte-for-byte from the sealed mechanism archive. Endpoints use original
raw source indices, including original coherence/revisit/local roles and
exclusions. Every required input must be accepted. Every existing pair must be
checkable, at least 90% within the original 1.5 nominal-mm tolerance, with no
loss above one percentage point on an originally good pair. Missing poses or
checks remain missing. Handheld acceptance is reported separately: those two
archives have no fixed geometry roster and cannot earn a geometry pass.

The old-adverse and both handheld projects use the original legacy loader,
zero skipped inputs and its original resolved 60..300 depth band. Three modern
raw captures use twelve skipped frames and explicit 60..300. The bounded wrapper
only translates the legacy scene flag when delegating the unchanged M21 tool;
it does not alter algorithm, source selection, crop or decision gates.

## Verified before changed trajectories

Direct and wrapped observation replays match exactly for modern A1 and all
188 legacy handheld inputs: full decisions, poses, offered/accepted indices,
solver traces, coherence, factors and input identities. Modern raw pair scores
also match; legacy geometry equality is explicitly unavailable, not invented.
The direct reference outputs were hash-bound before the comparisons.

All 51 new focused tests pass. The selected M1–M24 plus Stage B suite passes
1,231 tests, lint and cold typing of 125 files. Source is frozen in 164 executing
files and 252 separate verification files. These are implementation checks,
not fresh physical accuracy, handheld, GUI throughput or thermal qualification.

## Complete qualification result

| Recording | Accepted / required inputs | Weakest fixed agreement | Below-90 pair records | Good-pair regressions >1 point |
| --- | ---: | ---: | ---: | ---: |
| Slow D | 1,609 / 1,609 | 56.898% | 18 / 65 | 10 |
| Development extended | 1,171 / 1,171 | 35.011% | 24 / 103 | 24 |
| Old adverse | 2,703 / 2,703 | 1.186% | 33 / 165 | 42 |
| Later adverse | 2,359 / 2,359 | 7.776% | 20 / 147 | 27 |
| Handheld with pauses | 186 / 188 | No original geometry roster | — | — |
| Combined handheld | 93 / 303 | No original geometry roster | — | — |

Every longer recording fails both the 90% floor and the unchanged good-pair
regression guard. Pair records may share endpoints; these are not independent
observations. Paused handheld offers all 188 and refuses two; combined offers
96, accepts 93, then the original three-refusal stop leaves 207 unoffered.
Total: 8,121 accepted, five refused, 207 unoffered. Missing work is not excluded.
Original archived acceptance was 185 paused and 284 combined: one additional
paused acceptance does not offset the combined-motion failure.

All 164 executing source files still match saved/live hashes. The full rescore
reloads and verifies raw inputs and reproduces all score fields, not just the
verdict. There are 24,745 completed fit requests and seven refused requests,
all seven in combined handheld; 170 unconverged reverse seeds are recorded.
A completed process is not a complete accepted trajectory or quality pass.

The 04:49:13 host-only snapshot is GPU 48/50/72°C edge/junction/memory and CPU
package 75°C. Scanner temperature is unmeasured. The completed rejection permits
the documented M22 continuation; firmware, production integration and fresh
physical qualification do not follow from it.

## Earlier partial findings: 04:43 UTC

Development extended accepts all 1,171 inputs but fails: minimum fixed agreement
35.011%, with 24 below-90 pair records (some revisit records share endpoints).
Slow D accepts all 1,609 inputs but fails: minimum 56.898%, with 18 below-90
records across its original and sealed-D schedules. Remaining runs still execute;
these findings do not substitute for the complete six-case audit.

Development's worst pair is consecutive inputs 520/521: original 99.231%,
changed 35.011%. At frame 521 the chosen model candidate has plane residual
0.646, while the accepted adjacent candidate has 0.0667. Both clear the original
gates; a low local residual does not prove a correct long trajectory.

A complete read-only saved-alternative audit preserves all 103 development
pairs. Every one of its ten below-90 fixed-local pairs has an accepted adjacent
candidate preserving the full original pair guard. Only one of seven bad revisit
endpoints has any such saved alternative (the two revisit record kinds overlap).
Thus some immediate corrections were available, while later history was already
wrong across all saved options. This is a counterfactual at saved states, not
a new selection policy, closed-loop test or physical-truth claim.

The audit's initial ordinary-only candidate bound stopped at recovery-attached
decisions. Its original source and correction note are retained. The completed
audit includes original recovery candidates; no scan or fixed score was edited.

The corresponding complete 65-pair D alternative audit finds an accepted
guard-preserving saved candidate at only one of seven bad original-coherence
pairs and two of eleven bad sealed-D pairs. Most recorded late states cannot
be repaired simply by picking another already-saved candidate. An earlier
different selection history is not ruled out by that counterfactual.

Completed alternative audits on old/later adverse recordings retain all 165/147
fixed pairs. Their 18/4 below-90 fixed-local pairs all have an accepted adjacent
candidate preserving the original guard. Across the three local schedules,
that is all 32 failing local checks. No accepted alternative repairs old
adverse's four bad original-coherence or eleven bad prior-revisit records;
later adverse has one preserving alternative among fifteen bad prior revisits.
Those are saved-history observations, not a tested preference change.

## Handheld refusal diagnosis

All original and changed refusal rows are retained with exact raw-index mapping.
Paused refusals move from original indices 185/186/187 to changed 127/183;
combined changes from nineteen scattered refusals to 93/94/95 followed by stop.
Original archived default and changed grid/centered/reverse runs differ in
several settings; count differences do not isolate reverse initialization.

Combined 93/94 reject disagreeing candidates; 95 has no passing candidate.
Returns at those inputs are only 5.26%, 6.77%, 6.72% of the frame, against recent
baselines 15.06%, 14.31%, 13.43%. The advisory warning does not reject them;
actual disagreement/travel/overlap/fit rules do. Paused 127 also has a return
warning, while 183 does not. These observations establish co-occurrence, not
the physical reason for missing depth. Choosing a different preference *inside*
an agreeing group cannot resolve two disagreeing groups at the same saved state.

The completed archive is sealed at 2026-09-09 04:54:01 UTC: 516 files,
985,742,765 bytes, manifest SHA-256
`1b9daa8e20bd046d28a5be8ff569e916dd32f4abf5cd7cdec97b323ddceee38c`.
The exact inventory and every hash are independently rechecked outside the
archive at 04:55:14 UTC, in the M22 evidence root. These are local copies and
integrity checks, not off-machine backup or physical accuracy verification.
