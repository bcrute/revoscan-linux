# Reliability continuation: solver and model mechanisms

8 September 2026. In progress; no implementation is qualified for the live app.
Continues [the interaction campaign](reliability-factorial.md) and
[the reliability roadmap](reliability-roadmap.md).
Evidence: `/home/ben/scans/diagnostics/20260908-reliability-mechanisms/`.

That original root is sealed. Later phases have separate archives and are
tracked by the [living learning index](reliability-learning-log.md); do not
infer current status from an earlier phase checkpoint in this historical record.

This campaign tests distinct geometric mechanisms and selected combinations,
using the exact five F1 recordings and all 105 original required pair records.
Every candidate must accept every required input, reach at least 90% agreement within
1.5 nominal mm on every check, and preserve original good-control pairs within
one percentage point. Both fixed-camera views, both 20 ms controls and moving C
must pass before longer/handheld testing. Accepting frames alone is not success.

Production source, hardware settings and prior sealed evidence are unchanged.
All changes are offline tools/tests. Actual GPU normals are required in every
trajectory trial; concurrent replay timing does not measure GUI throughput.

## M1: centered finite rotation updates — rejected

M1 applies the existing linearized rigid update around its matched-point
centroid, avoiding camera-origin-dependent finite translation terms. It retains
the original correspondences, singular guard, trimming, residual and stopping.
Twenty trajectories cover the four predeclared bases on all five recordings.
All inputs are accepted in all twenty, but no whole configuration survives.

| Base with centered update | Static A minimum | Static B minimum | A1 minimum | A2 minimum | Moving C minimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stride / pooled / residual | 78.48% | 93.42% | 94.56% | 99.09% | 98.15% |
| Rank / pooled / settling | 97.63% | 99.78% | 29.77% | 95.09% | 16.00% |
| Grid / pooled / residual | 97.37% | 99.78% | 98.26% | 98.74% | 15.82% |
| Rank / native / residual | 99.24% | 99.78% | 94.50% | 43.95% | 20.80% |

Grid plus centering passes four case guards, but fails C badly. Stride preserves
C and A2, yet fails static A and the static-B/A1 regression guards. Minima alone
do not report all pair regressions; complete per-pair scores are retained.

Two instrumented-original pilots exactly reproduce full F1 decisions, poses,
input mapping and coherence. Across 19,443 fixed-correspondence shadow steps,
neither original nor centered update increases the fixed target-plane objective.
Their surface updates nevertheless differ by up to .2438 nominal mm on A1 and
.1507 on C's rank/settling base. This supports numerical/history sensitivity,
not a certified physical pose error or a demonstrated need for line search.

## M2: symmetric source/target normals — rejected

Own implementation of equations 10–11 in
[Rusinkiewicz's symmetric ICP paper](https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2019_ASO/symm_icp.pdf).
Both native source and target normal directions enter the fit. Missing source
normals remain missing and are counted before the unchanged trimming step.
Externally reported residuals and all tracker gates retain their original units.

All twenty planned trajectories complete. No whole configuration survives.
For example, grid's C minimum is 32.89%, still a clear failure despite improving
on that F1 base. Rank/native C drops to .4173%, while its static cases remain
above 99%. Rank/settling A1 and A2 each refuse one required input; full input
acceptance remains binding. Two further original pilots reproduce F1 exactly.

## Normal and model diagnostics

The fixed 288-cloud covariance audit finds a mathematical edge case: an exact
line can pass the current normal classifier even though its normal is not
unique. No near-collinear eligible neighborhood is found in the sampled real
clouds. Approximately 4–6% have a weak smallest/middle eigenvalue separation,
including the good static B recording. This is not a discriminating failure
detector and does not justify a tuned hard cutoff.

All forty earlier R1 model targets are reconstructed exactly with raw/provenance
hash checks. Choosing an actual voxel contributor instead of its centroid
changes coordinates by up to 1.1244 nominal mm. Median displacement is zero
in every target; bad targets do not show larger normal-direction displacement
than good ones. Coordinate differences motivate a bounded intervention, but
do not establish that averaging causes the tracking failure.

A further read-only audit reproduces all forty D query scores and inspects each
pre-offer keyframe archive. None of the fifteen bad queries has a comparable
nonlocal archived raw view under the existing descriptor rules; five of the
twenty-five good queries do. This is not proof that every noncomparable cloud
is useless, but it gives no support for re-ranking the existing archive alone.

An alternative raw-image-distance eviction metric is simulated at all 1,110
original insertion/anchor events, keeping twelve keyframes, the anchor and two
protected recent entries. It adds no comparable nonlocal match for the bad
queries and loses four of the good group's five matches. All original archive
events reproduce exactly. This fixed-event simulation does not fit poses or
model the changed insertions a real feedback policy would create; the simple
archive intervention is not advanced from these negative results.

## M3: confidence, directional balance and their combination — rejected

M3 tests source planarity weights, normal-direction balance, and both together;
its unweighted centered corner is the completed M1 result. All sixty changed
trajectories complete successfully as processes, but none of the twelve complete
configurations passes all five case guards. The best pass only three cases.
Every one of the 420 fixed-pair interaction contrasts is computable; no missing
value is imputed. Full constituents are retained in `m3/interactions.json`.

Interactions are large in both directions, not generally beneficial. On A1
source 9/529 with rank/pooled/settling, centering alone gives 29.77% agreement.
Planarity and directional balance individually each raise it to 99.74%, yet
combining them gives only 16.99%. Conversely, on C source 107/610 with stride,
the individual weights give 21.65% and 88.00%, whereas their combination reaches
98.50%, close to the centered corner's 99.32%. That same combined C trajectory
still has another required check at 82.94%. These are matched reused-data
comparisons, not independent accuracy measurements or per-recording switches.

## M4: measured model points and solver origin — rejected

M4 replaces model voxel centroids with their nearest actual contributing points,
preserving target count/order, source downsampling and either pooled or native
normal construction. Both original and centered updates are crossed with the
four bases. All forty changed trajectories complete; none survives all cases.
Three observation pilots reproduce F1 exactly, including the native-normal path.

Both grid variants pass four case guards, but C remains at 15.85% with the
original update and 29.14% with the centered update. Their static and A1/A2
minima remain above 97%. All 420 fixed-pair model-coordinate by solver-origin
contrasts are computable. This is a genuinely tested combination, not evidence
that combining other mechanisms can never work.

## M5: covariance-oriented local fitting — rejected

M5 is an explicitly limited, own-code experiment inspired by
[Generalized ICP](https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf).
It fits a three-dimensional residual with source/target normal-oriented
covariances. The fixed plane-thickness ratio .0004 comes from the
[authors' reference implementation](https://github.com/avsegal/gicp/blob/master/gicp.cpp).
It retains the existing ten-neighbor normals and takes one centered
Gauss–Newton step per outer iteration, rather than reproducing the paper's
twenty-neighbor covariance estimation and inner nonlinear optimizer. Original
plane-information refusals, external residual semantics and all gates remain.

All twenty changed trajectories complete, and the two observation pilots match
F1 exactly. No configuration passes more than three case guards. Grid A2 falls
to 8.37% and C to 17.20%, while its static checks stay above 99%. Accepting every
input in those recordings does not rescue the geometry. This rejects the
declared approximation, not every implementation of generalized ICP.

## M6: reciprocal correspondence matching — rejected

M6 keeps a forward nearest-point pair only when the reverse nearest-source
query chooses that same source point. Exact ties use original source order.
Removed correspondences count as rejected before the unchanged trim; no gate
is weakened. Both original and centered updates are crossed with all four bases.

All forty changed trajectories complete, with two exact baseline pilots and
420 computable fixed-pair interactions. None of the eight configurations passes
more than two case guards; the stride configurations pass none. Several
stationary/turning trials also refuse required frames. Even when every moving
input is accepted, geometric agreement can approach zero. The declared strict
reciprocal rule is rejected, not a general theorem about every matching method.

## Further read-only diagnostics

All 106,491 M1 recorded steps contain finite local-information eigenvalues;
none is missing and all smallest eigenvalues are positive. The matched-cloud-
radius-normalized condition ratios do not isolate bad trajectories. For example,
good grid/static-A has median 219.76, while failed stride/static-A has 91.65;
good grid/A1 reaches a maximum 679.45 while failed rank/settling A1 reaches
137.25. This gives no support for selecting a new singularity cutoff. Biased
normals/correspondences can still produce a full-rank fit, so this is not proof
of physically accurate pose information.

The five Stage A capture parents contain no saved RGB sequence or separate IR
channel. All fifteen inspection PNGs compare pixel-for-pixel with renderings of
their original raw depth frame. Color-assisted tracking cannot be replayed from
those images; another capture set would need synchronized image/depth evidence
and calibration/skew validation. No new acquisition was performed here.

An additional audit evaluates every original D candidate on all forty R1 checks.
At all fifteen bad checks, even the best eligible candidate remains below 90%.
A proposed unused-reference raw validator has nonempty retained reference pools,
but none of the forty pools supplies a comparable raw view after excluding all
candidate training references and protected validation endpoints. That ranking
idea is therefore untestable under this independent-reference construction,
not demonstrated accurate or inaccurate. The candidate oracle at a drifted
saved state does not rule out a different earlier causal history.

## Sampling and remaining work in this execution

A count-only audit of every original candidate on the twenty base/case cells
reproduces all source counts and model-target counts exactly. None of these
prepared frames exceeds the 900-point source cap. The ceiling-stride model
sampler nevertheless leaves usable capacity unspent: stride/static-B has a
median 1,203 model points and 597 unused available points. Stride/A1 leaves a
median 197 unused; grid/A1 leaves 104. Several static rank/grid bases instead
have too few available points to fill 1,800. This is not evidence of a GPU
optimization regression or a demonstrated cause of drift.

M7 doubles incoming/retained budgets to 1,800 on stride/grid pooled residual
bases; model target capacity and all quality gates stay unchanged. Three
900-point observation pilots reproduce F1 exactly. All twenty changed trials
complete, but none of the four whole configurations survives. Original-update
grid passes four cases, including all three moving controls: minima 99.16%,
98.91% and 98.66% on A1, A2 and C. Static A is 97.26%. Static B remains above
96.91%, yet five original pairs lose more than one percentage point, with a
maximum loss of 2.474 points. Its apparent median surface displacement from
identity reaches .651 nominal mm. This is a promising partial result, not a
reason to relax the stationary regression guard. All 210 capacity-by-origin
contrasts are computable.

M8 separately fills the existing ordinary-model allocation with evenly
stratified measured points, without changing source/retained/recovery sampling
or any point cap. Three observation pilots match F1 exactly. All twenty changed
trajectories complete, but none survives; original-update grid passes three
cases and its C minimum is 21.15%. All 210 sampling-by-origin contrasts are
computable. M8 uses a scoped adapter and hash-bound companion report; original
replay reports are not rewritten.

M9 explicitly combines the exact M7 capacity and M8 model stratification, with
the same two bases, two update rules and five recordings. This combination was
declared before the changed M8 results. Its three observation pilots reproduce
M7's dense original-update trajectories exactly. All twenty changed trajectories
complete, and all 420 four-corner capacity-by-sampling contrasts are computable.
No whole configuration survives. Centered/grid passes four cases: A1/A2/C
minima 98.86%, 98.69%, 98.84%, with all moving regression guards preserved.
Static A is 95.07%. Static B is 96.75% but loses as much as 2.974 percentage
points on an original pair. Original-update grid passes three cases, failing
the static-B and C regression guards despite minima above 96%.

Combination effects can genuinely be beneficial: on A2 source 113/625 with
centered grid, the original, capacity-only, sampling-only and combined corners
score 98.74%, 96.93%, 39.70% and 99.75%, respectively. That does not erase the
same configuration's stationary failure. All four corners, not only that
improvement, are retained in `m9/interactions.json`.

The shared M6/M7 executing snapshots contain 134 files, including the SDK JSON.
Earlier M4/M5 snapshots contain 131 Python files; their omitted SDK JSON is
preserved in the initial snapshot and independently unchanged. M8's executing
snapshot contains 136 files and M9's 138, both including the SDK JSON. No source
used by the M8 matrix was changed until all twenty processes had finished.

Read-only preparation checks do not support catastrophic filtering loss on
static B: grid prepares a mean 831.18 points at the 900-point target and 1,697.63
at the 1,800 target. Filtering follows thinning and estimates its spacing from
that cloud, rather than using a fixed neighborhood radius. These counts do not
identify the cause of the dense variant's remaining drift.

A print/CAD reference exists in the repository, but it cannot yet be treated
as ground truth for this recording set. The print's formal measurement gate is
false (missing repeatability/instrument/manufacturing metadata), and the prior
target record explicitly mentions a different later white model. Current raw
captures are not bound to the older print ID. Asset hashes and these limitations
are recorded without running CAD alignment, generating a mesh or changing poses.

A separate explicit real-GPU reference test passes. Fresh disjoint regressions
cover 698 passed/two skipped core tests and 336 passed GUI lifecycle tests.
The latest 659-test diagnostic integration suite passes, as does a cold type
check of 108 production/tool files. These overlap earlier focused suites and
are not added to them as if those were independent tests. M9's initial adapted
test fixture used the wrong companion filename/CLI option (six failures); the
fixture was corrected before any M9 real replay. The first type-check invocation
also omitted the local source root; the corrected invocation and an explicit
profile re-export pass without weakening typing rules.

## Additional discriminating checks and M10

Every adjacent and frame-zero/current raw comparison is checked on all five
recordings: 5,038 comparisons from 2,524 verified raw inputs. The existing
view-comparability rule accepts all stationary comparisons, but also 537/627,
553/627 and 563/627 adjacent pairs on A1/A2/C. A direct adjacent-view pose hold
would therefore confuse slow motion with stillness. Fixed-anchor comparisons
are more discriminating but still accept early motion and revisits. No pose
hold, new cutoff or inferred motion encoder is installed.

The recorded-cloud GPU parity audit completes after M9 and before changing
shared helpers for M10. All 424 normal fields, containing 470,140 estimates,
match the CPU reference exactly, including coordinates, normal components,
missingness, neighbor counts and field summaries. The 384 fixed raw-source
clouds cross grid/stride with 900/1800 capacity; all forty stored R1 model
targets are also checked. Their GPU fields additionally match the earlier R1
records exactly. This supports numerical parity on these samples, not physical
normal or pose accuracy and not a guarantee for every possible input.

A separate all-frame static audit identifies another sampling instability.
At the observed two-reference model allocation, dense native inputs preserve
about 97% of their pixel locations between adjacent frames. Ceiling-stride
model selection reduces median common/smaller-sample retention to about 49%;
M8's exact list strata preserve about 51%. Applying the existing S3 fixed pixel
ranks inside the model preserves 96.56% on A and 96.89% on B at exactly M8's
900-point allocation. Every frame and selected pixel list is retained. This
is fixed-allocation pixel evidence, not a changed model-membership trajectory.

M10 therefore tests native-pixel grid selection for ordinary model points,
preserving original point order/coordinates, reference frames/poses, normal
rules and all quality gates. Grid/pooled/residual input is crossed with the
existing 900 and M7's 1800 capacities and both solver updates: twenty changed
trajectories after six exact observation pilots. The 141-file executing source
snapshot includes the SDK JSON. Shared M8/M9 default behavior remains tested;
the new replay is separate and labels its counts as pixel-grid, not strata.

All twenty M10 trials complete; none survives. Both 900-point variants pass
only the stationary cases. Both dense variants pass static A, A1 and A2, but
static B falls to 83.51%/84.41% and C to 13.24%/14.79% (original/centered).
All 210 capacity-by-selection contrasts are computable. Improved pixel
stability therefore does not establish improved registration.

The pixel helper's first attempt incorrectly treated the calibration record as
an object with direct fx/cx attributes: 24 initial tests and the first diagnostic
failed. It now uses the repository's existing validated intrinsics accessor;
24 corrected tests and the complete two-case audit pass. The failed evidence is
preserved. No calibration value, pixel-rank rule or quality tolerance was tuned.

The subsequent model-density audit reconstructs 1,152 fixed queries across
twelve grid configurations. Sixty queries have no formed model; all 1,092
formed models reproduce their recorded construction, and full-union voxel
centroids are exactly idempotent under the existing reducer. Many models leave
hundreds of the 1,800 distinct-cell capacity unused after pre-budgeted points
are merged. This is not a catastrophic collapse unique to a failed variant,
and wrong saved poses can inflate available-cell counts. M11 tests voxelizing
the same retained reference union before applying the unchanged 1,800-point
model budget, at both capacities and update rules, with all original guards.

All 8,333 Stage B raw inputs have also been reread and match their original
manifests. Three sources are legacy projects and need their existing legacy
loader settings (no diagnostic-scene override, no depth-band override). This
is source readiness, not a changed Stage B trajectory or a reliability pass.

All 2,519 adjacent pairs are included in a separate arrival-timing diagnostic.
At zero lag, Pearson arrival-interval/raw-change-score correlations are -0.912
and -0.840 on the stationary controls and -0.869/-0.874/-0.871 on A1/A2/C;
all fixed -1/0/+1 lag results and rank-bin summaries are retained. Shorter
dequeue gaps therefore do not imply smaller raw changes even in static scenes.
These timestamps are host arrival times, and raw changes are not measured pose.
The result does not justify rescaling velocity by these intervals. The earlier
firmware record already documents two distinct depth deliveries per processing
cycle; this audit does not establish a new exposure-time or calibration model.

## M11/M12: separate spatial coverage from capacity components

M11's six observation pilots reproduce F1/M7 exactly. Its 143-file executing
snapshot is frozen before replay. The first unit tests caught a consumed
world-point iterator and empty-cloud reduction; both are fixed, and all seventy
focused tests pass before real replays. Centroids, selected indices and actual
reference contribution counts are recorded; a reported frame with no selected
voxel contribution is refused rather than counted as model support.

All twenty M11 changed trials complete, with no whole survivor. Both ordinary
variants pass the two stationary cases only. Both dense variants pass static
A/B and A1/A2, but C falls to 30.43%/30.42% (original/centered). In contrast to
M7/M9, static B stays at 99.78%, but that trades the earlier moving success for
a failed slow-turntable trajectory. All 210 capacity-by-voxel-first contrasts
are computable. More distinct model cells do not qualify placement accuracy.

M12 is separately declared before changed M11 results. It holds preparation at
1,800 and crosses fitting-source and retained-history caps of 900/1,800. M7
already supplies the both-dense corner; thirty new cells cover the remaining
three corners with grid/pooled/residual and both updates on every original case.
This isolates preparation, fitting and retention effects that M7 increased
together. The original model and every other policy field are unchanged.
A scoped offline settings adapter records the actual full policy in both the
original replay and a hash-bound companion, with no report rewriting.
Thirty-eight focused tests pass; a type-check caught reuse of a binary-stream
variable for text output, now corrected. The 145-file snapshot includes the SDK
JSON and is frozen before M12 pilots. M12 GPU runs wait for M11 to finish.

The CPU-only normal-support audit completes on all 192 fixed grid source clouds
and forty R1 model targets. It records both ten- and twenty-neighbor normal
fields, every support radius and angular difference. On stationary B, the mean
median radius is 4.57 nominal mm at 900 points/ten neighbors, 3.68 at dense/ten
and 4.96 at dense/twenty. This establishes changed support scale, not which
normal is physically correct. M13 separately declares one factor-of-two normal
support test at both capacities and update rules, behind M12 and any survivor's
Stage B work. Thirty focused tests pass, including exact explicit CPU-ICP
delegation and report/object/exception preservation. No production normal count
changes. A strict type check rejects using a non-exported imported tracking
symbol as a public attribute; the wrapper now names the original function from
its defining module and checks the internal hook explicitly.

Before M12 starts, the independent twenty-neighbor GPU parity audit checks all
232 support-scale clouds: 276,410 point estimates, every complete NormalField
equal to the CPU reference, and every saved twenty-neighbor field reproduced.
All 232 eigensolve dispatches run on the RX 9070 XT, with no numerical mismatch.
This supports implementing the declared scale experiment, not angular accuracy.
M13's 147-file snapshot includes all executing helpers and the SDK JSON.

The aggregate audit through M11 verifies 300 changed trajectories across sixty
whole configurations, 6,300/6,300 required pair checks and 34 exact reproduction
pilots. No configuration survives. All completed runs report GPU activation and
none reports a fallback. These are repeated development recordings, not 300
independent physical scans or a measured success-rate population.

M12's six baseline observation pilots reproduce F1/M7 exactly. A derivative
static pixel audit then holds each proposed retained sample to exactly its
original ceiling-sampler count, checking the corresponding M12 prepared counts
on all 640 inputs. Median adjacent common/smaller pixel retention is
48.873%/48.878% for original sampling and 97.034%/97.384% for fixed pixel ranks
on A/B. All 638 adjacent pairs are retained. This separates point identity from
the count increase in M10's model experiment; it neither changes a trajectory
nor establishes a new retention policy's geometric reliability.

All thirty M12 changed trials complete, with no survivor. All six configurations
fail static B's original regression guard. Preparation 1,800/source 900/retention
1,800 with the original update passes the other four cases: weakest agreements
are 96.51% on static A, 98.52%/98.98% on A1/A2 and 97.62% on C. Static B reaches
95.58%, but loses up to 3.48 percentage points on an originally good pair.
Both preparation-only configurations also accept only 319/320 static A inputs.
The source-1,800/retention-900 variants pass static A and A1/A2 but regress on
both static B and C. All 210 fixed-preparation source-by-retention contrasts
are computable. A high minimum alone does not waive the regression or acceptance
requirements.

M13's six ten-neighbor observation pilots reproduce F1/M7 exactly. All twenty
changed trajectories complete; none survives. Both dense variants pass four
cases but fail static B's regression guard: weakest agreement 96.62%/96.73%,
maximum loss 2.714/2.710 percentage points (original/centered). The ordinary
original variant also passes four cases, but A2 falls to 9.17%. Ordinary
centering passes three; A2 falls to 5.12% and C regresses by up to 4.25 points.
All 210 fixed density-by-normal-support contrasts are computable. Larger support
does not qualify this tracker, even where three moving controls now pass.

M14 separately declares retained native-pixel selection crossed with normal
support at fixed preparation/source 1,800 and retention 900. The proposed
retained sample has exactly the old ceiling sampler's count on every admission,
keeps the original native coordinates and point order, and changes no other
tracker admission behavior. Its three new retention/support corners, crossed
with both updates and all five recordings, require thirty changed trials; M12
supplies the original-retention/ten-neighbor corner. Three exact M12 observation
pilots must pass first. The 149-file executing snapshot is frozen, thirty-three
focused tests pass, and actual admission counts, selected-index hashes, fit
normal counts and underlying report hashes are recorded. M14 waits for M13
and yields priority to Stage B if any whole configuration survives.

All three M14 observation pilots reproduce M12 exactly. All thirty changed
trials complete, with no survivor. Pixel-selected retention/ten-neighbor normals
with centering passes four cases, but static B reaches only 90.80% and loses
up to 7.779 percentage points. The original update passes just static A/A2.
Both pixel-selected/twenty-neighbor variants pass static A/A1/A2, but C falls
to 39.96%/11.44% and static B to 89.20%/89.77% (original/centered). The original
retention/twenty-neighbor variants pass one/zero cases and each refuse one
static A input. All 210 fixed retention-by-support contrasts are computable.

M15 then declares the missing density-by-existing-S1 stopping interaction, at
grid/pooled/ten-neighbor normals with original and centered updates. It changes
only the closed offline capacity adapter's permitted stopping factor; the S1
implementation, thresholds, solver steps, model and every acceptance/recovery
rule remain unchanged. F1 supplies the original-capacity/original-update/S1
corner, leaving fifteen new trajectories. Nine exact reproduction pilots cover
the residual controls at both capacities and the reused S1 corner before any
changed replay. Forty-five focused tests and the four existing S1 tests pass,
including actual numerical solver composition, with a separate cold eighty-file
type check. The 151-file executing source snapshot is frozen. This experiment
waits behind M13/M14, with any survivor's Stage B work taking priority.

All nine M15 observation pilots now reproduce the corresponding F1/M7 results
exactly, including the old grid/settling A1 refusal (627/628 accepted). The
fifteen changed density/stopping trajectories are complete; none of the three
whole configurations survives. Ordinary/centered passes only the two stationary
cases; A1 loses 1.554 points, A2 and C each refuse one input, and C falls to
22.626%. Dense/original passes static A and A1/A2, but static B loses 2.654 points
and C falls to 31.123%. Dense/centered passes static A, A1 and C, but static B
loses 2.598 points and A2 loses 1.426 points. All 315 required pair checks and
210 density-by-stopping contrasts are computable. Observation-mode reproduction
of a refusal does not waive full acceptance for a new candidate.

The read-only Stage B pair-reproduction audit verifies all 480 previously
recorded saved-placement comparisons exactly, including covalid point counts
and median depth offsets, plus all forty D raw-identity reports. The records
are D 65, development 103, old-adverse 165 and later-adverse 147; overlapping
checks and original fit/validation roles are preserved. The handheld sources
have no original usable revisit/local pairs, and no replacement checks are
invented. Legacy 75% coherent flags remain historical, not the 90% promotion
margin. This is endpoint/arithmetic readiness, not corrected Stage B geometry.

## M16: rotation-aware covariance descent — rejected

A complete read-only M5 step audit finds 108,077 available cost records. The
frozen-metric objective increases once, by 7.1e-15; the rotated-metric objective
increases 8,093 times, up to a ratio of 1.086922. This happens in both passing
and failing cases, so it is not a useful failure classifier by itself.

M16 differentiates the rotation-dependent covariance in Generalized ICP's
equation 2, using the original positive frozen-metric preconditioner for a
centered descent direction. A fixed sixteen-trial Armijo backtrack evaluates
the actual rotated objective, explicitly refusing if it cannot find a step.
It preserves M5's original plane-information refusal, source-normal eligibility,
trimming, external residual/stopping and every tracker gate. It is not the full
paper optimizer or a physical-accuracy guarantee. All trial costs, derivatives,
accepted steps and refusals remain recorded.

Forty-six focused tests pass: analytic/finite-difference gradients at identity
and nonidentity, known-transform convergence, sign/magnitude invariance,
coordinate equivariance, forced backtracking/exhaustion, inherited normal and
residual behavior, and replay report/factory/scope guards. A type check caught
an implicitly imported experiment class; its import now names the defining
module, and the corrected eighty-file check passes. The 153-file executing
snapshot is frozen. Four exact M5 observation pilots, including full solver
records, precede twenty changed trials on the original four bases and five
cases. All four controls reproduce M5 exactly and all twenty changed trials
complete, but none of the four whole configurations survives. Grid passes the
two stationary cases and C (98.9523%), while A1/A2 fall to 7.9181% / 6.2363%.
Stride passes only static B and C; A1/A2 lose 7.225 / 1.085 points and static A
falls to 83.5355%. Both rank variants pass only the stationary cases; rank/S1
also refuses one A2 input. All 420 original pair checks and M5 comparisons are
computable. Per-step objective descent does not ensure trajectory correctness.
Execution is in `/home/ben/scans/diagnostics/20260909-reliability-covariance-descent/`:
the original archive approached its unchanged 15 GB integrity-manifest bound.
This location-only change preceded every M16 trajectory; all 153 copied source
hashes, the protocol and F1 plan must match the original preparation exactly.

The complementary M11/M13 four-case outcomes also motivate a direct
[voxel-model plus normal-support combination](reliability-voxel-normal-combination.md).
M17 completes six exact observation controls and twenty changed trajectories;
none of its four configurations survives. Every input is accepted, but moving
raw-agreement minima range down to zero. All 420 matched interaction contrasts
are retained. The distinct [M18 voxel-pyramid test](reliability-multiscale.md)
completes six exact controls and twenty changed trajectories with no survivor.
Ordinary density fails C; dense configurations regress on previously good
controls. The aggregate through M18 is 455 changed trajectories, 91 whole
configurations, 9,555 checkable pairs and 74 exact controls. See the
[living learning index](reliability-learning-log.md) for subsequent status;
any survivor has Stage B priority and nothing is qualified.

The aggregate through M14 verifies 380 changed trajectories across 76 whole
configurations, all 7,980 required pair records checkable and 49 exact
reproduction pilots. M15 adds fifteen trajectories, three configurations,
315 pair checks and nine exact controls. No configuration survives. Completed runs report
actual GPU activation with no fallback messages. The fourth source-boundary
check confirms all 79 production files and unrelated initial owner files remain
unchanged; only the two opt-in replay runners differ among initial files.

The twenty-trajectory F1/M1/M7 grid prediction audit finds no motion prior in
any stationary-control frame. Moving runs record priors on 147–239 frames each,
but no saved motion-prior seed and no candidate whose travel gate relies on its
prediction allowance. Every non-anchor frame has a locally accepted adjacent
edge, which supplies the keyframe/model seed. This makes a no-prediction
ablation a low-priority explanation for these particular outcomes, not a claim
about all future handheld motion. An initial audit used the wrong seed literal;
its artifact remains, and the corrected audit imports the production constant.
No velocity timing adjustment or pose-hold rule is installed.

The completed M1–M15 archive is sealed: 5,235 files, 14,168,620,484 bytes,
manifest SHA-256 `2f17db2b758157a93deb78e060a96ace1384d40795dcc242609bb9df7b7d0823`.
Its exact inventory and every hash verify from the M17 continuation archive.
M16 and later results are separate; the seal is integrity, not qualification.

The subsequent raw-pair support audit includes all 7,350 M1-M13 checks. Across
the 3,163 pair records belonging to passing individual cases, the minimum
covalid-vote count relative to the same original baseline pair is 95.285%;
the median is 101.126%. Minimum ratios are 99.084%/99.503% on static A/B,
95.285%/97.927% on A1/A2 and 97.235% on C. Among all pairs reaching 90%
agreement, including otherwise rejected cases, the minimum is 85.393%.
Every record is retained. No new support cutoff is fitted, no case verdict is
changed, and covalid votes are not a measurement of whole-object coverage.

The four bases are a selected interaction roster, not every possible
combination. All are development replays of existing recordings. Longer scans,
handheld coverage, live integration and scanner thermal qualification still
require a geometry-preserving survivor and the subsequent roadmap gates.
