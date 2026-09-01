# The depth-completeness benchmark

Status: added 2026-08-29. This started as **checkpoint 1** of the
depth-completeness thread, the oracle the later spatial and temporal
completion tasks have to satisfy, written before them and implementing no
completion of its own, so that a denser result cannot be called an
improvement when it bridges a depth edge, closes a slot, moves a dimension,
roughens a plane, invents geometry in empty space, degrades tracking, or
quietly relabels a reconstruction as a measurement. **Checkpoint 2** (spatial
completion) and **checkpoint 3** (pose-compensated temporal completion) have
since been added to the same corpus and are scored by the same oracle,
gates and refusal vocabulary described below.

The contract lives in
[`tests/depth_completeness.py`](../tests/depth_completeness.py): what a
completion must say about every sample, what is refused outright, and what is
measured about what it said. The scenes live on disk as data under
[`tests/fixtures/depth_completeness/cases/`](../tests/fixtures/depth_completeness/cases),
and one command runs them. The rules are proved in Python by
[`tests/test_depth_completeness.py`](../tests/test_depth_completeness.py), and
the harness itself - that corrupting a fixture, a scene, an expectation or a
manifest makes it fail rather than quietly measure something else - by
[`tests/test_depth_completeness_harness.py`](../tests/test_depth_completeness_harness.py).
pytest runs both, and `.github/workflows/ci.yml` runs the command itself as its
own job — see [In CI](#in-ci).

## The one command

```bash
.venv/bin/python -m tests.depth_completeness_harness --out depth-completeness-report.json
```

- **`0`** — every candidate of every case produced exactly the outcome its
  fixture declares.
- **`1`** — the harness ran and at least one did not. An unexpected pass, a
  missing expected failure, an extra failure, **a corruption control that was
  not refused**, a malformed fixture and an exception raised while scoring are
  all this.
- **`2`** — the harness could not run at all: no corpus, a manifest that is not
  a manifest, a case file that is not where the manifest says, an output path a
  report may not be written to, or an artifact that could not be published.

Cases are independent and every one is attempted, so a failing artifact
preserves all of them rather than the first.

On the machine this was written on the run takes about **7.5 s** on CPU, with
no GPU, no display server, no network and no device: the compute backends are
the pure-Python references, the corpus is committed data, and **two runs over
an unchanged corpus are byte-identical** — no duration, host or path appears
anywhere in the artifact. Today's corpus exits **0** with 10 of 10 cases and
106 of 106 candidates behaving exactly as declared.

Regenerate the corpus, when the scenes themselves change, with:

```bash
.venv/bin/python -m tests.build_depth_completeness_fixtures
```

That module is the reviewable recipe behind every committed number. It is a
command, not a test; pytest never runs it, and a run that changes a fixture is
changing evidence.

### Why this is a second command beside `tests.quality_harness`

Both harnesses measure through the same `scanner3d.quality` primitives, use the
same gate factories, and publish through that module's own path-safety and
serialisation code, which this one **imports rather than repeats** — so there is
one authority for how a fixture is read and how an artifact is published. What
differs is the evidence. A quality-harness case *is* its geometry, written out
as reviewable JSON; a completeness case is a family of 512000-byte raw depth
frames plus per-sample dropout masks and exact poses, which is not reviewable
as committed JSON. And a quality-harness case declares one expected outcome,
where a completeness case runs many candidates against one set of gates,
because the whole point is that the same bar refuses a dense invented fill and
admits a supported one. Folding either shape into the other would have meant a
new schema version of the accepted quality artifact and a case kind its reader
has no schema for.

## In CI

`.github/workflows/ci.yml` runs the command above as its own job,
`Depth-completeness benchmark`, on public GitHub-hosted `ubuntu-latest` with
`contents: read`, a bounded timeout, CPython 3.14 from the same pinned
`setup-python`, and the same `uv sync --frozen --extra dev` as every other job.
Every action is pinned to a reviewed full commit SHA. The step runs the exact
command this page documents and nothing wraps its exit status, so **1** (a
candidate that did not do what its fixture declares) and **2** (a corpus that
could not be run at all) both fail the job.

The whole report goes to the step log — every case, every candidate, every
expected-versus-observed gate, passing ones included — so a failure can be read
without downloading anything. The upload step is guarded by `if: always()` with
`if-no-files-found: error`, because the failing run is the one whose report is
worth keeping and a report that was never written is a fact worth failing over.
The artifact name is `depth-completeness-report`, distinct from the
reconstruction-quality job's `quality-harness-report` and from the surface
oracle's `surface-oracle-report`, so one run leaves three separately named
reports rather than a collision.

**Hosted and CPU-only on purpose.** This is a correctness oracle over committed
synthetic truth run through the pure-Python compute backends: a GPU or the
owner's LAN/heavy machine would change how fast the answer arrives, not what
the answer is. `tests/test_ci_workflow.py` parses the workflow offline and
asserts all of the above — the runner, the read-only permissions, the pinned
action identities, the frozen sync, the exact command, that no step or job
swallows a failure, that each job uploads the file its own command wrote under
its own name, and that nothing anywhere reaches for a self-hosted runner, an
accelerator, hardware or a deployment path. It never dispatches Actions and
never uses the network.

It is the **second** of **three** named CPU correctness-oracle jobs, beside
`Reconstruction quality harness`
([docs/quality-harness.md](quality-harness.md)) and `Surface regression oracle`
([docs/surface-oracle.md](surface-oracle.md)). They are separate jobs because
they measure different evidence and keep different artifacts, so a
depth-completion regression stays legible as itself rather than as "the harness
went red". Ordinary pytest still runs unchanged in its own job, and it runs this
corpus too, through the module-scoped fixture in
[`tests/test_depth_completeness_harness.py`](../tests/test_depth_completeness_harness.py);
this job is additional *named* evidence with its own status and its own retained
report, not a replacement for the suite and not a gap in it.

## Provenance: four states, and support that can be checked

Every sample of the scored window carries one of exactly four states, and a
state nobody defined is a claim nobody checked:

| State | Means |
| --- | --- |
| `measured` | the camera returned this sample, and the value is byte-for-byte the raw one |
| `spatially_reconstructed` | filled from evidence **in this frame** |
| `temporally_reconstructed` | filled from evidence **in another frame** |
| `missing` | not returned, not filled, and carrying no depth |

Two things are carried beside the state rather than folded into it.
**Confidence** is separate, and is a finite fraction of 1: a value that is not
a measurement does not become one by being confident. **Support** is named, one
measured sample of one frame, spelled `frame:<i>:pixel:<u>,<v>`, and *every*
reconstructed sample must name at least one. The benchmark then resolves those
names against the input frames, so "supported" is checked rather than believed:
support at a pixel that returned nothing is a refusal, not a passing gate.

A `temporally_reconstructed` sample that stands behind an accepted spatial
estimate carries one more fact beside its state: a **confirmation verdict**,
published for every sample of a run that was offered an accepted spatial
estimate to check. It is one of `confirmed`, `contradicted` or
`not_observed`, plus `no_spatial_estimate` where the spatial stage proposed
nothing to check at all. `confirmed` and `contradicted` are the two verdicts
reachable only once independent posed evidence has settled the sample for
itself — which is why **both** carry `temporally_reconstructed` provenance,
not just the one that agreed with the spatial guess. A contradicted sample
publishes the *independent* posed value, never the spatial one and never an
average of the two: a value halfway between two disagreeing surfaces is
evidence of neither. `not_observed` is different in kind — no other posed
frame measured the sample at all — so it keeps its spatial provenance, its
spatial code and its spatial confidence, unchanged; passing through a
confirmation stage that had nothing to check is not itself a confirmation.

A candidate may also declare a **work record** — what its run cost and what it
refused, by reason. It is optional, because the hand-written controls here have
no run to report, and it is the only part of a candidate's account that is
about the run rather than about the output. Every field of it is therefore
bound to something the benchmark counts for itself; see
[the work record](#the-work-record-and-the-confidence-distribution).

### The seventeen refusals

`tests/depth_completeness.py` declares seventeen refusals — `frame_set_mismatch`,
`raw_frame_mutated`, `sample_count_mismatch`, `unknown_state`,
`confidence_out_of_range`, `measured_value_changed`, `missing_carries_depth`,
`reconstructed_without_depth`, `unsupported_reconstruction`,
`support_not_measured`, `support_without_reconstruction`,
`temporal_support_not_another_frame`, `spatial_support_not_this_frame`,
`depth_code_out_of_range`, `work_counts_inconsistent`, `unknown_confirmation`,
`confirmation_inconsistent`.

A refusal is **not** a gate failure. A refused result is not scored at all,
because a metric computed from an account of the data that does not hold would
be a number about nothing. A fixture that expects a refusal names the exact
refusal code it must earn, so "refused for some reason" is never the refusal a
control was written for.

## The window, and why every denominator is fixed

Each case is scored over one rectangle of one 640x400 frame, and every
denominator in the benchmark is the number of samples in that window times the
number of frames. That is not convenience. A metric whose denominator is
"whatever the candidate looked at" can be improved by looking at less, and no
gate built on one is evidence. A result that returns fewer samples than the
window cannot be scored at all (`sample_count_mismatch`), and a result that
declares its hard samples missing moves a fraction *down* rather than out of
the arithmetic.

## The ten cases

| Case | Scene | Exists to catch |
| --- | --- | --- |
| `plane-fronto-complete` | a fronto-parallel plane, **no dropout at all** | the benchmark measuring its own noise: on complete input an identity completion must reproduce every measured value and every downstream metric exactly, and a candidate that discards measurements to smooth the result must still fail the gates that notice |
| `plane-fronto` | the same plane with speckle, a connected hole and a one-sided strip collapse | an oracle that refuses **density** rather than invention. Constant depth is the one geometry nearest-neighbour completion reproduces exactly, so a dense fill is *right* here — and the same corpus refuses the same candidate on every other geometry |
| `plane-oblique` | a tilted plane, same three dropout shapes | a surface known to be flat coming out lumpier than it went in. Tilt is the whole difference from the case above |
| `curved-surface` | a sphere against empty space, speckle and a connected hole | two failures a plane cannot show: a smooth curve a staircase fill roughens, and real empty space around the silhouette that a dense fill paints with whatever was nearest. False fill is gated at exactly zero here |
| `depth-step` | a box in front of a backing plane, dropout hugging the depth edge | the failure the whole thread exists to prevent: dropout is worst exactly where geometry is most discontinuous, so a completion is most tempted to interpolate precisely where interpolation is meaningless |
| `narrow-slot` | two boxes with a three-column gap showing the backing plane, the gap dropped | closing a gap that is really there. It makes the result denser, smoother and more plausible, and the object is now the wrong shape. Reconstructing the *background* inside the slot is the right answer; putting foreground depth there is not |
| `aperture-ambiguity` | four coplanar slabs at a constant depth with **one** genuinely empty sample in them, paired with **one** held-out dropout twenty columns away | the observation ambiguity the slot case evades by being large. Both holes present byte-identical 9x9 input evidence, one is surface and one is a hole in the object, and no single-frame method can answer both. The case exists so that whichever side a completion loses is declared rather than hidden behind a size cap |
| `multiview-object` | an asymmetric object from three exact poses, with complementary intermittent loss | temporal completion's whole claim — that another frame saw what this one missed — which is true only after the pose between them is applied. This is where carrying the same pixel across motion is proved to fail while carrying it through the exact poses does not |
| `temporal-motion` | one box in front of a backing plane, seen from three laterally displaced cameras, each missing a complementary strip of columns, with the camera moving exactly five pixels at the box's own depth between frames | the same failure as `multiview-object`, isolated down to one silhouette in motion: carrying another frame's sample at the *same* pixel puts backing-plane depth on a sample whose truth is the box front face — a 5,632-code error on the wrong surface — while carrying the same raw sample through the exact pose lands it where it belongs |
| `temporal-confirmation` | one genuinely empty aperture, one held-out plane sample and one held-out sample of a one-sample-wide pillar standing in front of the plane, from three posed views | the case a spatial-only stage cannot answer at all: the three holes present byte-identical evidence from inside one frame, and independent posed evidence has to reach three different verdicts about them — confirmed, contradicted, and not observed |

Three dropout shapes rather than one percentage is deliberate throughout: the
mask kinds are isolated speckle, a connected interior hole, dropout hugging a
depth edge, a one-sided strip collapse, a region that never returns at all, and
complementary intermittent loss across posed frames. One overall dropout
percentage would hide every one of those differences behind one number.

Two details of `multiview-object` are load-bearing. The complementary masks drop
a different third of the columns in each frame, which is the opportunity a
temporal method exists to take — but **one block of columns is dropped in every
frame**, and no temporal method can recover it. And the worst-sample error gate
is deliberately absent there: reprojecting a real sample through exact poses
rounds it across the silhouette by one pixel, and a worst-sample gate would
refuse the honest control for a rounding artefact. The single-view cases gate
the worst sample.

## The work record, and the confidence distribution

Two measurements were added at checkpoint 2, when the first real completion
arrived. Both exist so that a completion's *own* claims about itself are
checkable rather than decorative.

`confidence` is computed **here**, from the result and the truth, and never
taken from the candidate: it is what a candidate's confidence values actually
predicted. It carries the whole distribution as five fixed bins whose
populations add up to the reconstructions coverage counted, the subset at or
above the fixed threshold of **0.5** together with the worst error any of those
confident reconstructions actually made, and the mean confidence on real
surface beside the mean in empty space. `confidence.high_confidence_worst` is
then gated at the same 16-code worst-sample tolerance the held-out error gate
uses. A completion is free to be unsure; what it may not do is be sure and
wrong, and lowering its own confidence to leave the gate is visible in the same
measurement.

`completion_work` is declared **by** the candidate — the deliberately
forgeable half — and every count in it is bound to something the oracle
computes: the holes are the input's own no-return samples, the attempted count
is the holes, the acceptances are the reconstructions the result carries, the
support total is the support it named, the reasons add up to the refusals, and
the budget is one neighbourhood scan per hole at the policy it publishes.
`completion_work.within_budget` is gated, so bounded work is a stated fact
rather than a wall-clock measurement. The policy is published beside the counts
because that is what makes an accepted sample reproducible from its own named
support.

Two more measurement kinds arrived with checkpoint 3, and each is present or
absent by **candidate identity**, not by the case: `depth_completeness_temporal_work`
(published as `temporal_work`) is present for every candidate in
`CANDIDATES_WITH_TEMPORAL_WORK` — the pose-compensated temporal stage and its
four corruption controls — and carries the same accounted-for shape as
`completion_work`, plus `confirmed`, `contradicted` and `not_observed` counts,
gated the same way at `temporal_work.within_budget`.
`depth_completeness_confirmation` (published as `confirmation`) is narrower
still, present only for `CANDIDATES_WITH_CONFIRMATIONS` — the two-stage
candidate and the control that forges a confirmation — because it is only
meaningful once there is an accepted spatial estimate for independent posed
evidence to check. It carries the three confirmation verdicts, each split by
whether the sample actually sits on real surface, and two of its own fields
are gated absolutely: `confirmation.confirmed_false` at exactly zero, because
a confirmed sample standing in space the scene does not have is not what
"confirmed" can mean, and `confirmation.confirmed_on_surface`, the count a
confirmed verdict is supposed to be right about.

## Provenance-weighted fusion

`depth_completeness_weighted_fusion` (published as `weighted_fusion`, with
`weighted_fusion_surface` beside it) is what the accepted
[weighted TSDF input contract](weighted-tsdf-input.md) does with a candidate's
own frames: the same frames again, through
`scanner3d.tsdf_input`'s adapters and
`scanner3d.tsdf.TsdfVolume.integrate_weighted`, into a volume of the same
configuration the `fusion` record uses.

The two records are a pair, and collapsing them would leave nothing to compare.
`fusion` is what uniform fusion does with the candidate's completed codes —
provenance ignored entirely — and is republished inside the weighted record as
`unweighted_observed_voxels` and `unweighted_surface_vertices` so the
comparison survives into a case that does not carry both. Beside them
`matches_measured_only` says whether the weighted field is byte-identical,
values and accumulated weights both, to the field the same frames' raw bytes
produce through the same entry point.

That same raw multi-view field is also measured for what it puts out, not only
compared byte for byte. `measured_only_observed_voxels`,
`measured_only_surface_vertices` and `measured_only_surface_coverage` are its
output measures — same frames, same poses, same volume, same extractor, same
truth model — and `observed_voxel_gain`, `surface_vertex_gain` and
`surface_coverage_gain` are the weighted field's minus those. Those three are
the only geometric claim available in the record; `voxel_updates` and
`contribution` are accounting and are not one. `measured_only_frames` is the
frame count that baseline was given, and the strict reader requires it to be
every frame the record was offered — so a gain measured against a baseline with
the supporting views withheld is refused rather than read.

`withheld_support_frames`, `withheld_support_observed_voxels` and
`withheld_support_weighted_observed_voxels` are that forbidden comparison,
computed on purpose and named for what it is: one frame fused alone, raw and
weighted. They carry no gate.

Only `multiview-object` and `aperture-ambiguity` fuse a weighted volume, and
they carry it for opposite reasons: on the first the accepted temporal result
must reach the field **without costing** the raw baseline anything, and on the
second the accepted spatial result must reach **exactly as much** of it as the
measurements alone do. No case gates a gain — see the multi-view negative
result above.

The route into it is closed on purpose. A weighted volume is reachable only
through the accepted `CompletedPatch` or `TemporalPatch` a candidate carries in
`CandidateResult.accepted` — each of which has already bound itself to its own
immutable raw input and re-derived its own run — and a candidate's published
states, codes and confidences are refused unless they *are* that result. A
candidate with no accepted result publishes no weighted-fusion measurement at
all, and the case records its weighted gates as inapplicable rather than
letting a self-reported confidence past them. `dense_nearest` fills the
aperture at confidence 1.0 and could just as easily fill it at 0.1; it has no
accepted result to offer, so it has no weight to be low.

## The candidates are controls, and one algorithm

`CANDIDATES` in `tests/depth_completeness.py` holds twenty-seven named
functions. Nine of the twelve below are controls, each existing to make one
gate prove it is load-bearing. `spatial_completion`, `temporal_completion` and
`spatial_then_temporal_completion` are the exceptions: they are product code.

| Candidate | What it does |
| --- | --- |
| `baseline` | the unchanged pipeline: what returned is measured, the rest stays missing |
| `identity` | completes nothing, run on complete input — must be exactly the baseline |
| `dense_nearest` | fills every hole with its nearest returned sample |
| `broad_interpolation` | averages the nearest measured sample along each of the four axes |
| `foreground_bias` | prefers the nearer surface among nearby samples — a heuristic that sounds careful |
| `naive_temporal_carry` | carries another frame's sample at the **same pixel**, ignoring the motion |
| `posed_temporal_carry` | carries another frame's samples **through the exact poses** |
| `easy_samples_only` | fills only holes with close support and leaves the hard ones missing |
| `discarded_measurements` | throws away the extreme returned samples, then fills densely |
| `spatial_completion` | **a product primitive** — `scanner3d.depth_completion` over each frame's window on its own, through a thin adapter that recomputes nothing |
| `temporal_completion` | **a product primitive** — `scanner3d.depth_temporal` reconstructing a sample from another posed frame's own measurement, with no spatial estimate offered to check |
| `spatial_then_temporal_completion` | **a product primitive** — the same temporal stage, given `spatial_completion`'s accepted estimates to confirm, contradict or leave not observed |

Every value `foreground_bias` writes was measured somewhere nearby, so nothing
about its provenance is false; what is wrong is where it put it. That is the
point of having it.

Beside those, **nine corruption controls**, each of which must earn one exact
refusal — a run in which any of them is scored at all is a run whose oracle has
stopped working:

| Control | Must earn |
| --- | --- |
| `forged_measured_states` | `measured_value_changed` |
| `mutated_raw_frame` | `raw_frame_mutated` |
| `unsupported_fill` | `unsupported_reconstruction` |
| `forged_confidence` | `confidence_out_of_range` |
| `missing_carrying_depth` | `missing_carries_depth` |
| `unknown_provenance_state` | `unknown_state` |
| `truncated_window` | `sample_count_mismatch` |
| `phantom_support` | `support_not_measured` |
| `temporal_support_from_own_frame` | `temporal_support_not_another_frame` |

And **six controls that corrupt a real completion result** rather than a
hand-written fill, which is what makes them controls on the product code:

| Control | Corrupts | Must earn |
| --- | --- | --- |
| `spatial_completion_forged_work` | reports one refusal as an acceptance, changing nothing else | `work_counts_inconsistent` |
| `spatial_completion_recursive_support` | cites one of its own reconstructions as the evidence for another | `support_not_measured` |
| `temporal_completion_forged_work` | reports one refusal as an acceptance in the temporal work record | `work_counts_inconsistent` |
| `temporal_completion_self_support` | cites a sample of its own frame as the evidence for a temporal fill | `temporal_support_not_another_frame` |
| `temporal_completion_unmeasured_support` | cites a pixel that returned nothing as the evidence for a temporal fill | `support_not_measured` |
| `temporal_completion_forged_confirmation` | calls a spatial estimate confirmed when no other posed frame observed it, and leaves it spatial | `confirmation_inconsistent` |

## Results worth stating

All of these are verified by the committed corpus at exit 0.

- **`dense_nearest` passes on the fronto-parallel plane and fails on every
  other geometry it is run against.** On constant depth, nearest-neighbour is
  the right answer and the oracle says so. On the oblique plane it fails both
  roughness gates and both error gates; on the sphere it fails false fill,
  error, dimensions, landmarks and observed-to-reference agreement; at the
  depth step it fails `edge.misplaced` and the worst-sample error gate; in the
  slot it closes 72 samples and fails `slot.open`; in the multi-view case it
  fails false fill, error and fused-surface agreement, and the counterfactual
  replay shows it would also move the tracker if it were ever fed to one.
  **The oracle refuses invention, not density.**
- **The unchanged baseline fails exactly the acceptance-target gate**
  (`coverage.supported_fraction`) in every masked case, and **additionally the
  step-column gate in `depth-step`**, because dropout along the edge displaces
  the apparent boundary by four columns against a one-column tolerance. That is
  a real defect and the one a completion is supposed to repair, not an accident
  of the fixture. The baseline's reconstructed coverage is zero and its false
  fill is zero: it invents nothing, and its missing coverage stays visible as
  missing rather than being scored as a success.
- **`posed_temporal_carry` reaches accurate depth and still fails the
  acceptance target.** Its p95 error is one raw code, inside the four-code
  tolerance, and it clears dimensions, fused-surface agreement and tracking —
  and it still misses `coverage.supported_fraction`, because one column block
  is dropped in every frame and no temporal method can recover it. It is
  declared to fail rather than being called a success.
- **`naive_temporal_carry` fails on error and on fused-surface agreement**
  beside the coverage target, and is the worst candidate in the corpus in the
  counterfactual tracking replay. Same idea, same frames, without the pose.
  This is the "naive carry across motion fails" control, and it is what makes
  `posed_temporal_carry` mean something.
- **Every metric is denominator-bound to the whole window**, so
  `easy_samples_only` — which fills only holes with three close neighbours —
  scores well on error where it acts and fails the coverage target anyway. A
  result cannot report a superb error by never attempting anything difficult.
- **Provenance weighting keeps unconfirmed guesses out of the field, and buys
  no geometry on this corpus.** On `multiview-object`, `temporal_completion`
  offers 16 temporally reconstructed samples worth 5.32 of fusion weight
  between them and makes 5953 weighted voxel updates against the raw
  baseline's 5728 — and the field it produces has the same 3099 observed
  voxels, the same 333 surface vertices and the same truth-referenced coverage
  as the same three raw frames fused without it. `observed_voxel_gain`,
  `surface_vertex_gain` and `surface_coverage_gain` are all exactly zero and
  are gated for non-inferiority only; there is no gain gate, because an update
  count is not a coverage or surface measure. The `withheld_support_*` fields
  show what withholding the supporting views would have made this look like —
  1949 observed voxels raw against 2174 weighted, on one frame alone — which is
  why the negative result is the honest one. On `aperture-ambiguity` the two
  accepted single-frame estimates contribute exactly `0.0`, the weighted field
  is byte-identical to the measurements' own field, and the uniform
  counterfactual beside it observes 8457 voxels against 8445 and pulls out 897
  surface vertices against 893 — which is the aperture being filled, and
  exactly what the weighting refuses. See
  [docs/weighted-tsdf-input.md](weighted-tsdf-input.md).
- **There are two tracking replays per candidate, and only one of them is an
  acceptance route.** `tracking` and `tracking_pose` are the *intended
  dataflow*: `scanner3d.tracking` over each frame's own measured samples, with
  nothing a completion added. That is the only route this project's tracker
  has, and it is what every tracking gate is stated over. Because no completion
  enters it, this record is a property of the scene rather than of the
  candidate: on `multiview-object` every candidate reproduces the baseline's
  worst placement error of 1.09 degrees and 0.155 depth units exactly, and the
  strict reader **refuses** a report in which two candidates disagree about it.
  That also closes a hole: the cloud is built from the candidate's own output,
  so a candidate that damaged a measurement it was told to preserve moves this
  record and is caught.
- **`tracking_completed_cloud` and `tracking_completed_cloud_pose` are the
  counterfactual, and they carry no gate.** They replay the same tracker over
  the candidate's *completed* cloud and answer a question this project does not
  act on: what would happen if completion output were fed to the tracker.
  Against the scene's exact poses the answer on `multiview-object` is that it
  degrades placement — baseline 1.09 degrees, `posed_temporal_carry` 1.21,
  `temporal_completion` 2.38, `dense_nearest` 8.53, `naive_temporal_carry`
  13.49 — while every candidate still *places* all three frames, so a
  count-only record would have called them all neutral. They are published
  because that answer is unfavourable and would otherwise be invisible, and
  they are ungated because a bar over a route the product does not take would
  read as permission to take it. No candidate is claimed to have improved
  tracking on either route.
- **`temporal_completion` recovers exactly what another posed view really
  measured on `multiview-object`, and still fails the coverage target.** It
  recovers the 16 samples another posed frame actually saw, every one within a
  single raw code of the truth, invents nothing, and leaves the measured-only
  tracking record exactly the baseline's — which is the acceptance condition,
  because the tracker consumes measured geometry only and this stage does not
  feed it. In the counterfactual replay it moves the tracker's placement of
  frame 2 from 1.09 to 2.38 degrees, and that measurement is the concrete
  reason temporal output enters no tracker, no mesh, no export and no live
  preview in this task; it reaches a fused volume only through the
  provenance-weighted route, at a weight derived from its own accepted
  confidence. It fails `coverage.supported_fraction` alone.
  `spatial_then_temporal_completion` behaves identically there, because the
  spatial stage recovers nothing when every hole is a whole column.
- **On `temporal-motion`, `temporal_completion` passes every error, edge,
  step-column, dimension, confidence, work and recovery gate, and fails only
  `coverage.supported_fraction`.** The speckle holes that are in every frame
  are in no other posed view either, so no temporal method can recover them,
  and the fixture states that as the expected outcome rather than as a
  surprise.
- **On `temporal-confirmation`, `spatial_then_temporal_completion` still
  fails `coverage.false_fill` and `confidence.high_confidence_false_fill`.**
  The accepted single-frame aperture ambiguity from `aperture-ambiguity` is
  disclosed rather than laundered by a second stage: a spatial estimate that
  no other posed frame observed stays `not_observed`, keeps its spatial
  provenance, and the false fill it already carried is still false fill.
- **`discarded_measurements` fails the measured-coverage floor everywhere it
  appears, and the named dimension and landmark gates besides**, including on
  `plane-fronto-complete`, where there is nothing to interpolate and the
  candidate can only destroy measurements. It is a *scored gate failure* and
  not a refusal: throwing away returned samples is a result the benchmark can
  read and judge, so the corpus judges it. A refusal is reserved for a result
  whose own account of itself does not hold, which is why relabelling a
  reconstruction as measured is refused and discarding it is scored.
- **`spatial_completion`, the product primitive, gains strictly on both planes
  and invents nothing anywhere.** It clears `coverage.local_recovery` on all
  four cases that state one — 48, 50, 54 and 34 owed samples — while the
  baseline recovers none of them; its false fill is exactly zero on every
  geometry one frame can settle; it bridges and misplaces nothing at the depth
  step, leaves the slot entirely alone, and returns `multiview-object`
  unchanged so the fused surface, both tracking records, the poses and the
  dimensions are the baseline's exactly. It **fails
  `coverage.supported_fraction` in every masked case**, and
  `edge_step_column.worst` at the depth step, and the fixtures declare those
  failures rather than calling conservative refusal a success.
- **`aperture-ambiguity` is where it loses, declared rather than tuned away.**
  It recovers the paired dropout and invents a surface across the true
  aperture, failing `coverage.false_fill` (1 against 0) and the case's own
  `confidence.high_confidence_false_fill` (1 against 0) - confidently, because
  the evidence behind the two samples is byte-identical and confidence is
  derived from that evidence. No threshold separates them; nothing in one frame
  does. The fixture states both failures as the expected outcome.
  [docs/depth-completion.md](depth-completion.md) has the per-case numbers and
  names every hole that remains; `temporal-confirmation` carries the same
  ambiguity into three posed views, and
  [docs/depth-temporal.md](depth-temporal.md) has that module's own account
  of it.
- **The confidence gate has teeth.** `confidence.high_confidence_worst` fails
  `dense_nearest`, `broad_interpolation`, `foreground_bias` and
  `easy_samples_only` on the geometries where they publish a confident and
  wrong sample — the curved silhouette, the depth step and the slot — while
  `spatial_completion`'s confident reconstructions are right to within the
  truth's own one-code quantisation.

## Where the limits come from

Never from what a candidate happened to score. Every limit is one of three
kinds, and each gate's `context` in the artifact says which:

1. **The acceptance target.** `coverage.supported_fraction` requires that
   **0.95 of the surface the scene really has** ends up measured, or
   reconstructed onto surface that is really there. It is expressed as a
   fraction of the whole window, so in a case whose window is mostly empty
   space the written limit is 0.95 times the scene's own truth-surface fraction
   — 0.814427 in `multiview-object`, 0.525798 in `curved-surface`, and 0.95
   where the surface fills the window. The unchanged baseline does not reach it
   in any masked case, which is the whole reason this thread exists.
2. **Non-inferiority bars**: the baseline's own value plus a stated margin in
   raw codes — 4 codes for plane roughness, 8 for dimensions, landmarks and
   both directions of surface agreement (single-view and fused), and 8 codes
   for the tracker's worst placement offset beside 0.25 degrees for its worst
   placement rotation. That 0.25 is a sixteenth of the 4-degree rotation
   between adjacent frames in the only case that replays tracking: a fraction
   of the scene's own motion rather than a number picked to suit a candidate. These are the
   gates that say a completion may not make a plane rougher, an object smaller,
   a feature move, or a surface sit further from truth than not completing at
   all did. `coverage.measured_fraction` is the same idea with no margin: the
   raw measurements the sensor returned must all still be there and still be
   called measurements.
3. **Absolute tolerances in raw codes**, for error against held-out truth and
   for geometry that is exact by construction: error p95 **4 codes**, worst
   **16 codes**, the worst error of any reconstruction published at confidence
   0.5 or above **16 codes**, edge bridging and edge misplacement **exactly
   zero**, false fill **exactly zero**, the step column within **1 column**,
   and the slot **open**. In `aperture-ambiguity` alone there is one more:
   `confidence.high_confidence_false_fill` **exactly zero**. Elsewhere it would
   be vacuous, because false fill of any confidence already fails outright;
   there it is the load-bearing bar, because the two paired samples score
   byte-identical evidence and so cannot be separated by confidence at all.
4. **A strict gain the scene states about itself.** `coverage.local_recovery`
   requires that at least as many samples end up reconstructed onto real
   surface as there were dropped samples with a **returned measurement on all
   four sides of them**. That number is derived from the rendered truth and the
   masks — 48 on `plane-fronto`, 50 on `plane-oblique`, 54 on `depth-step`, 34
   on `narrow-slot` — never from what any candidate managed, so the bar cannot
   drift towards whatever an implementation happens to reach. Beside
   `coverage.measured_fraction`, which pins the measured population, it is a
   strict supported-coverage gain over the unchanged baseline, which recovers
   none of them.

   It is deliberately **not** applied to `curved-surface` or
   `multiview-object`. On a sphere, "surrounded by measurements" does not mean
   the surface between them is locally determined — two samples either side of
   a silhouette's shoulder bracket a surface turning away from the camera, and
   the bar would be demanding an error. In the multi-view case every hole is a
   whole column and no sample is surrounded at all.

5. **A scene-derived floor for temporal recovery.** `coverage.temporal_recovery`
   requires a floor of samples that at least two *other* posed frames really
   measured, with the correspondence between frames holding in both
   directions, excluding samples the scene puts on its own declared
   foreground boundary — again derived from the scene and the masks, never
   from what a candidate returns. The only count it may be read from is
   `temporally_reconstructed_on_surface`, the subset of
   `reconstructed_on_surface` that actually carries temporal provenance: a
   refusal, a contradiction and a spatial fill are each evidence *about* a
   sample rather than a sample put back, and none of the three may be counted
   towards this floor. It applies wherever a temporal candidate is scored —
   `multiview-object`, `temporal-motion` and `temporal-confirmation`.

Error is scored **only** on samples the candidate reconstructed where truth was
held out. Copying the input therefore cannot earn a good score, and samples
reconstructed into empty space are excluded from it and counted as false fill
instead, where they cannot be averaged away.

**Rounding cannot flip the baseline's own verdict.** Written limits are rounded
to six decimals, `>=` bars **down** and `<=` bars **up**, because rounding the
other way would fail the baseline against a value it actually reached — a
fixture bug that looks exactly like a regression.

**One allowance exists, and it is stated.** `multiview-object` gates false fill
at a fraction of 0.10 rather than at zero, because reprojecting a real sample
through exact poses rounds it one pixel outside the silhouette, and the case
would otherwise refuse the only honest temporal control it has. The curved
single-view case gates false fill at **exactly zero**, and that is where
invention is refused outright.

## Whose numbers these are

The benchmark owns only bookkeeping — which sample is which, against which
denominator, and what a completion said about where its value came from.
Everything measured belongs to an existing authority:

| Authority | Owns |
| --- | --- |
| `scanner3d.simulate` | renders the truth frames |
| `scanner3d.depth_metric` | decodes every frame, and owns the rule that a raw sample of zero is **no measurement** and never a point at the origin |
| `scanner3d.quality` | every distribution, gate, plane fit, symmetric cloud agreement, named dimension and landmark |
| `scanner3d.tsdf` / `scanner3d.tsdf_surface` | fuse the completed frames and extract the surface, unchanged, at their public API and the `python` backend |
| `scanner3d.tracking` | places the frames — twice: once from measured samples only, which is the intended dataflow and the only gated route, and once over the completed cloud as an ungated counterfactual. `scanner3d.quality.pose_sequence_error` then compares where it placed them against the scene's exact poses |
| `tests/quality_harness.py` | the strict-JSON reading and the atomic publishing, so there is **one** such authority and not two sets of rules to keep true |

Nothing in `scanner3d` imports the benchmark.

Depth is `raw code x depth_scale` in each scene's own constructed units, which
measure nothing physical. The depth scale is a power of two, so `code x scale`
is exact in binary floating point and a round trip through the uint16 frame
loses nothing beyond the quantisation the benchmark measures on purpose.

## Fixture immutability

A case does **not** commit its frames — a family of 512000-byte frames is not
reviewable JSON. What it commits is the scene (shapes, intrinsics, depth scale,
window, and each frame's pose as an axis, an angle and a centre of rotation),
the dropout masks by shape, the named seed, and the **SHA-256 and returned-sample
count of every truth frame and every masked input frame** those produce. The
harness re-renders from the scene and **refuses any case whose frames do not
hash to what the fixture says**.

So a change to the renderer, to a mask rule or to the scene shows up as a
refusal rather than as a quietly different benchmark, and the frames stay
immutable evidence without being unreadable committed data. The masks
themselves come from SHA-256 of `seed:block` rather than from a module RNG, so
the corpus does not depend on a generator implementation that is free to change
between Python releases.

Beside that, the ordinary corpus rules: the manifest records each case file's
size, SHA-256 and case id; a file the manifest does not name is refused,
because a case file nobody listed is a case nobody reviewed; and a fixture that
has been edited is no longer the evidence that was reviewed.

## What the artifact is held to when it is read back

`tests.depth_completeness_harness.loads_report` is the reader, and it makes no
authenticity claim: the artifact is not signed, and nothing here recomputes a
scene's geometry from it. What it does is refuse a report that contradicts
itself, which is what the report carries enough evidence to do.

- **Every published scene field is a field of that kind.** The window is inside
  the 640x400 frame and its sample count is its own area; the depth scale and
  both focal lengths are finite and positive; the region flags are booleans,
  the foreground code is a raw code, the slot columns are distinct columns of
  this window, and a fusion volume encloses something and states a depth range
  a measurement can fall inside. A boolean is never accepted where a number
  belongs.
- **Every pose is a pose.** Each frame's pose is held to
  `scanner3d.geometry`'s own schema and rebuilt through `Rigid.of`, so a
  reflection or a scaled matrix is refused, and both derived magnitudes are
  recomputed from the matrix and the vector beside them.
- **Every declared gate is the gate that was evaluated.** A gate declaration
  states a comparison, a limit of the kind that comparison takes, a unit and a
  context; each evaluated gate must reproduce all four, in addition to
  resolving to the measurement and path the case declared. A limit loosened
  after the run leaves the evaluated gate intact, and that is exactly the edit
  this refuses.
- **The counts the report states twice must agree.** The coverage domain is the
  window's samples times the frames; its truth, measured and dropout counts are
  the sums of what the scene's own frames say they return and lose; the
  reconstructed-error block's population and denominator are coverage's
  reconstructions and coverage's dropout. Fusion and both tracking replays were
  offered the scene's frames, and a pose sequence compares one pose per frame,
  so all of those counts are the published frame count. Which measurements exist
  at all is decided by the scene's regions, so a candidate cannot quietly stop
  measuring the slot, the fusion or either tracking replay it was scored over.
  And the measured-only tracking records are the *case's* fact, not the
  candidate's: every candidate that publishes them must publish exactly the
  same ones, because the tracker there is offered measured geometry only and no
  completion enters it.
- **The optional measurements exist exactly when the evidence beside them says
  they do.** Six blocks are legitimately absent for some candidates, and
  "legitimately absent" is the hole a whole population could be dropped
  through, so presence is derived rather than believed: `reconstructed_depth`
  exists exactly when coverage places reconstructions on real surface,
  `confidence` exactly when coverage counts any reconstruction at all,
  `fusion_surface` exactly when fusion extracted a surface with vertices,
  `tracking_pose` and `tracking_completed_cloud_pose` exactly when their own
  replay accepted every frame of a sequence of at
  least two, `completion_work` exactly when the candidate is one of the
  closed set that runs the spatial product primitive at all, `temporal_work`
  exactly when the candidate is in `CANDIDATES_WITH_TEMPORAL_WORK`, and
  `confirmation` exactly when the candidate is in
  `CANDIDATES_WITH_CONFIRMATIONS`. Deleting a measurement together with its gates and bindings is the
  whole-population form of the hard-sample omission attack, and it leaves the
  coverage record next to it still saying the population was not empty.
- **The confidence distribution covers every reconstruction, and its bins add
  up.** Its three population counts are coverage's own reconstruction, on-
  surface and false-fill counts; the five bins carry the fixed edges and
  partition the whole population; the confident subset cannot outnumber the
  population it is drawn from; the surface advantage must be the two means
  beside it, and zero when either population is empty; and a worst confident
  error cannot be stated with no confident reconstruction on surface to have
  made it. The threshold itself is the oracle's fixed 0.5 and not the
  candidate's, and the confident counts are bounded by the bins that lie wholly
  one side of it — a reconstruction in a bin entirely at or above the line is
  confident whatever the summary says. Summarising fewer of the hardest samples,
  or moving the line they are judged at, is how an ordering is improved without
  improving, and none of those routes is open.
- **The work record is held to the run it describes.** Its holes are the
  domain minus the measurements the input returned, its acceptances are
  coverage's reconstructions, its reasons add up to its refusals and come from
  the fixed vocabulary, its spend is inside its budget, and its budget is one
  neighbourhood scan per hole at the radius it publishes — not a number the
  candidate states. Its published policy is rebuilt through
  `scanner3d.depth_completion.CompletionPolicy`, so there is one authority for
  what a policy is, and the support total must be inside what that policy
  allows for the acceptances beside it.

  **Whether a work record exists at all is derived too**, from the candidate's
  own identity — which is a closed field of the report, checked against
  `CANDIDATES`. A scored candidate that runs the product primitive must carry
  `completion_work`, and one that never ran it must not. That closes the
  deletion attack: removing the block together with its gate and its binding,
  moving the gate into both inapplicable sets and recomputing the bookkeeping
  used to leave a report that still validated, and now does not. Both
  directions have a control, and a further test runs every candidate to check
  that the reader's closed set is still exactly the set that produces a work
  record — so the two definitions cannot drift apart.
- **Coverage cannot claim more of a population than the population holds.** The
  per-sample states partition two fixed sets - the input the candidate was
  given and the truth behind it - so the counts are held to the inequalities
  that follow: the truth surface is exactly what survived the dropout mask plus
  what it took; measured samples cannot outnumber the raw samples that came
  back; and the on-surface reconstructions, the supported total and the false
  fill cannot outnumber the truth surface or the truth empty space they are
  drawn from. Adding up internally is not enough on its own.

## What this proves, and what the recorded scan only suggests

**Proven here:** the oracle's own behaviour over synthetic truth. That the
provenance contract refuses each of the seventeen ways a result can misdescribe
itself; that the gates separate a supported fill from an invented one on ten
named geometries, two of which are deliberately unanswerable from inside one
frame and say so; that a dense fill is admitted where it is right and refused
everywhere else; that carrying a sample across motion without its pose is
refused while carrying it through the exact pose is not; and that the
unchanged baseline sits where the corpus says it sits. Nothing more.

**Only suggested, by one read-only recorded capture.**
`/home/ben/scans/scan/20260828-110641` holds 166 immutable frames, and reading
it gives:

| Observation | Value |
| --- | --- |
| Median single-frame coverage | 18.72% |
| Median 9-frame pixel union | 61.12% |
| Median 17-frame pixel union | 74.02% |
| Pixel locations never returning in the whole sweep | 11.54% |

Beside that, a live Qt depth-panel observation of 2026-08-29: 117,099 of
256,000 samples returned (45.7%), and the gaps are **raw code zero** rather than
lost Qt pixels — the samples are absent from the sensor's own output, not from
the display path.

Those are observations of one capture on one machine, not ground truth. The
mask shapes in this corpus are **patterned after** them; they are not that
capture, and **no number in this benchmark is a measurement of the scanner.**

Stated plainly, and written into every artifact this harness publishes: the
scenes are rendered by `scanner3d.simulate`, which is not a sensor model — no
speckle physics, no multipath, no subsurface scattering, no baseline shadowing.
A candidate that passes here has not been shown to work on the device. Every
temporal candidate here — `posed_temporal_carry`, `temporal_completion` and
`spatial_then_temporal_completion` alike — is given the scene's own exact
poses, which a real capture does not have for free, so this corpus measures
how much complementary evidence posed frames contain and not that any
algorithm can recover it, and the confirmation verdicts are only as good as
those same exact poses. Nothing in this artifact is a physical measurement of
the scanner. The tracking numbers are a
**record of what the unchanged tracker did** — with each frame's measured
samples on the gated route, and with the completed cloud on the ungated
counterfactual one; this benchmark
establishes no improvement in tracking and does not claim one. And this
benchmark validates nothing about the scanner.

## Acceptance gates for the next tasks

Checkpoint 2 added the first real completion, `scanner3d.depth_completion`, and
it does **not** yet meet gate 1 below on any masked case. That is recorded in
the fixtures as a declared failure per case rather than absorbed as a success;
see [docs/depth-completion.md](depth-completion.md) for exactly which holes
remain and why each one needs evidence a single frame does not have.

Checkpoint 3 added `scanner3d.depth_temporal`, and on `multiview-object` it
does **not** yet meet gate 1 either, for the same structural reason as
`posed_temporal_carry`: one block of columns is dropped in every frame, so no
temporal method can recover it, and `coverage.supported_fraction` still
fails. It meets gates 2 through 5 there — it recovers exactly the samples
another posed view really measured, every one within a single raw code of
the truth, invents nothing, names support in another frame for every one
of them, and leaves the measured-only tracking record exactly the baseline's.
That last one is met because the tracker consumes measured geometry only and
this stage does not feed it; the ungated counterfactual replay shows why that
boundary is where it is — feeding those accurate reconstructions into ICP over
a cloud this sparse moves where the tracker places frame 2, from 1.09 to 2.38
degrees — which is why temporal output is not yet wired into the
tracker, the fused volume, the mesh or the live preview. See
[docs/depth-temporal.md](depth-temporal.md) for that module's own account of
what it does and does not yet meet.

A spatial or temporal completion is accepted by this benchmark when it does all
of the following, on the committed corpus, at exit 0:

1. **Meets the acceptance target** — `coverage.supported_fraction` — without
   failing any non-inferiority bar or any absolute gate. Coverage bought by
   bridging an edge, closing a slot, moving a dimension, roughening a plane or
   painting empty space is not coverage.
2. **Keeps every raw sample byte-identical** and every measured value
   unchanged. The raw measurement is the evidence, and a completion may not
   edit it or discard awkward samples and interpolate over them.
3. **Names support for every reconstructed sample**, resolvable to a measured
   sample of a real frame — this frame for a spatial fill, another frame for a
   temporal one.
4. **Carries confidence separately** from provenance, as a finite fraction of 1.
5. **Keeps the measured-only tracking record at least as good as the
   baseline's**: the same frames placed, no new refusal, and — because every
   bad candidate here still places every frame — the same poses, within a
   sixteenth of the rotation between adjacent frames, measured against the
   scene's exact poses. This is the route the product actually takes: the
   tracker is offered measured geometry only. The ungated
   `tracking_completed_cloud` records beside it measure what feeding a
   completion to the tracker would cost, and are **not** an acceptance route —
   a candidate does not become acceptable by doing well there, and passing this
   list is not permission to wire completion output into the tracker.

Two things this list does not permit anyone to say. **Completed depth is not
camera measurement**, and must never be described, exported or displayed as
one; that is what the four provenance states are for. And **passing this
benchmark is necessary, not sufficient** — physical evidence is a separate,
later gate, and this artifact does not contain any.
