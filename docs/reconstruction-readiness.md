# Reconstruction readiness: what has to be true before a scan is published

Status: added 2026-08-29. Load-bearing. Every entry point that turns a capture
into a model — the window's **Reconstruct**, `scanner3d reconstruct`, and
`scanner3d.reconstruction_export.export_reconstruction` itself — now refuses a
reconstruction that fails any of the gates below, and a refusal leaves nothing
on disk.

Everything lives in
[`src/scanner3d/reconstruction_readiness.py`](../src/scanner3d/reconstruction_readiness.py)
with its regression cases in
[`tests/test_reconstruction_readiness.py`](../tests/test_reconstruction_readiness.py).

## Why this exists

On 2026-08-28 a capture kept writing frames for two minutes after live
alignment had stopped placing any of them. The folder afterwards held 176
frames, every one of which looked on disk exactly like a frame that had
aligned. [`docs/tracking-loss.md`](tracking-loss.md) is the answer to that at
the *capture* boundary: a loss is recorded, the folder explains itself, and
`scanner3d.capture_sequence.read_capture_sequence` refuses a capture whose loss
nothing closed.

This is the answer at the *reconstruction* boundary, and it exists because
three more failures have the same shape — a result that looks exactly like a
result:

- **The poses do not describe a scan.** Forty frames from one place register
  perfectly against each other. Held-out closure error is a genuinely honest
  check and it is *literally unchanged* by a trajectory that never moved,
  because there is nothing for a drifting trajectory to drift against.
- **The occupied surface was seen from one angle.** A million points is not
  evidence. A million points from one direction know one silhouette, and the
  shape that comes out of the other side is a guess.
- **The extracted surface disagrees with the points it came from.** A
  displaced, floating or grossly lumpy TSDF surface is topologically flawless,
  is watertight if you ask it to be, and opens in any viewer.

## What is measured, and by what

There is deliberately no single score. Four failures, four instruments, and one
result that carries **all** of them — because stopping at the first failed
check sends an operator round a loop: fix the poses, run it again, be told
about the coverage, run it again.

| Gate | Reason code | Catches | Blind to |
| --- | --- | --- | --- |
| Tracking | `tracking_lost`, `tracking_loss_record_unusable` | a capture whose durable loss nothing closed, reached through a *direct* API | a capture that lost tracking and never recorded it |
| Pose diversity | `pose_viewpoints_insufficient`, `pose_excursion_insufficient`, `pose_motion_below_noise` | identical, near-stationary, and stuck or substantially repeated trajectories | a pose the registration got wrong and was confident about |
| Coverage | `coverage_unsupported` | occupied surface seen once, or only from one direction | anything that was missed entirely — unseen volume is not counted |
| Surface agreement | `surface_unsupported_by_evidence`, `surface_abandons_evidence` | a fused surface displaced, floating or lumpy relative to its own points | a uniform scale error; anything below about two truncations |

Every reason code has exactly one piece of advice, in
`reconstruction_readiness.ADVICE`, and both front ends render that mapping
rather than each writing their own words — so the window and the command line
cannot drift apart about what an operator should do.

### Tracking: reach, not a second opinion

`read_capture_sequence` remains the sole authority on what closes a tracking
loss — the byte binding, the excluded-frame digests, the segment ordering. None
of it is duplicated here and none of it is weakened. What this gate adds is
**reach**: `export_reconstruction` takes a capture directory and a list of
poses, and it used to believe both. A caller handed clouds and poses from
somewhere else never asked the capture anything. Now it does.

The cheap path is preserved exactly. A capture holding no `tracking-loss.json`
costs one `stat` per run and never opens the authority at all. Only when a loss
record exists is `read_capture_sequence` called, and then its refusal is quoted
rather than paraphrased.

A **validated recovery is history, not a refusal.** A capture that lost
tracking and genuinely found the object again is usable, and a gate that
refused every folder mentioning a loss would throw away exactly the captures
the recovery mechanism was built to save.

### Pose diversity: three questions, because one number cannot say which

All three use the calibrated vocabulary that already exists —
`scanner3d.pipeline.frame_travel`, `NOISE_FLOOR` and `TRAVEL_MIN_MULTIPLE` —
rather than a new one, and none of them uses object identity or exact float
equality. A fixed probe of 128 points, taken by stride from the first non-empty
cloud, stands in for the object.

- **Adjacent travel.** How far the probe surface moves between consecutive
  frames, which is exactly the quantity the live tracker gates on. A step below
  the noise floor did not move.
- **Excursion.** The extent of each probe point's whole track, not the distance
  between the first and last pose. That distinction is load-bearing: an orbit
  that comes back to its starting view is how a trajectory earns a revisit
  constraint, and measuring endpoints would refuse the best scans here.
- **Distinct viewpoints.** Counted greedily against a separation of one useful
  scanning step, capped at 16. This is the instrument that catches a trajectory
  stuck between two places — travel is healthy, the excursion is real, and the
  sweep still knows nothing about the object it did not see.

Ordinary pauses cost nothing: the moving-step requirement is a small absolute
count rather than a fraction, so a scan is judged on the motion it has and not
on the proportion of its frames taken while standing still.

A uniformly scaled world is judged by the same noise floor, and that is the
honest answer rather than a gap. The floor is a property of the *scanner*, in
the units its depth samples are in, so a sweep scaled down until every step is
under it really did not move enough to see round anything.

### Coverage: occupied cells only

`scanner3d.coverage.coverage_map` does the measuring, which is the whole reason
it is used instead of a grid over the bounding box: unseen empty volume is not a
surface defect and must never manufacture a refusal. A cell counts as supported
when at least two distinct frames put a point in it and the widest angle between
their view directions is at least 20°. Half the occupied cells must be supported.

The cell comes from the placed geometry's own bounding diagonal, divided by 32 —
**not** from the fusion voxel size, which is a cost setting. Lowering a voxel
budget to make an export finish sooner must not change what counts as adequate
coverage. The cell is floored at twice the clouds' own estimated point spacing,
because a cell finer than the sampling measures the sampling: a cell holding one
point can only ever have been seen once.

### Surface agreement: internal consistency, explicitly

Both directions, always. Surface-to-evidence is blind to a surface that
abandoned half the scan; evidence-to-surface is blind to geometry that should
not be there at all. Reporting either alone passes a reconstruction the other
fails.

**Against the evidence the volume was actually fused from.** An export holds
two point sets and only one of them is the surface's own evidence: the
registered cloud is the decimated, filtered per-frame sample the registration
was fitted to, while the volume integrated the full-resolution `raw_frames`.
`fusion_evidence_sample` deprojects a bounded, deterministic sample of *those*
frames — the export's exact intrinsics and depth scale, the same inclusive
accepted depth interval, the same poses through the same `upright_pose` the
mesh and cloud are written in — and a zero sample is skipped as "no depth
returned", which is also how a support mask reaches this function. Nothing is
re-fused: this projects pixels to points and builds no field. The record
carries the frames offered, the frames sampled, the candidate samples examined
and the points that produced, so a reader can see what the fractions are
fractions of.

That distinction is load-bearing rather than tidy. Comparing the surface with
the registration sample means a defect that lives in the frames but misses the
decimation — most of every frame, by construction — passes unseen;
`test_a_defect_only_the_full_resolution_frames_can_see_does_not_escape` builds
exactly that capture and asserts both halves: the export refuses it, and the
registration sample the gate used to be handed reports nothing wrong.

**"Within" is inclusive.** A point at exactly the tolerance is supported, which
is the boundary `quality.cloud_agreement` already publishes for this
measurement. Distances are still clamped at the tolerance, because the search
does not look further, so a recorded distance equal to it means "this far or
further" and says nothing about whether the point was counted.

The tolerance is derived, never typed in: the larger of the volume's declared
truncation, twice the noise floor, and one diagonal of the cell both clouds were
thinned to. That last term is not optional — a measurement that thinned its own
clouds without widening its tolerance would be measuring the thinning.

Two things this is **not**:

- **Not accuracy.** Both clouds live in the reconstruction's own coordinate
  system, so a uniformly scaled world is perfectly consistent in it. Nothing
  here has seen an independent measurement of the object. See
  [`docs/quality-harness.md`](quality-harness.md) for what would.
- **Not a watertightness requirement.** A scan of one side of an object
  legitimately produces an open surface with real boundaries and passes.
  Nothing in this gate counts a boundary edge or asks whether the mesh closes.

Its resolution is stated rather than implied: it sees a displacement of about
two truncations and not one of half a truncation. It is a gross consistency
check by construction, and the record says which tolerance was used so nobody
has to infer it.

## Where the gates run, and what a refusal leaves

Inside `export_reconstruction`, at two points:

1. After the fusion resolution is chosen and **before a byte of field is
   allocated** — tracking, poses and coverage. A capture that fails any of them
   is refused before several minutes of projecting full-resolution frames into
   a volume nobody may keep.
2. After the surface is extracted and **before the output name is reserved** —
   surface agreement. Because nothing has been created yet at that point, a
   refusal has nothing to clean up: no `reconstruction-NNNN` directory, no
   `.reconstruction-partial-*` staging directory, no cloud, no mesh, no
   manifest. The export's existing atomicity is unchanged and unweakened.

They live in `export_reconstruction` rather than in each caller for the reason
above: it is the direct API, and a gate a caller can skip by not calling it is
not a gate. `scanner3d reconstruct` writes no geometry, so it calls
`evaluate_readiness` itself — and does so *before* `--poses-out`, so a refused
run produces no file of any kind.

A **published** record always carries a ready readiness under
`reconstruction.json`'s `"readiness"` key: the policy values, the measured
evidence and the decision. It is the evidence for why those files were allowed
to exist, not a warning beside them.

## The record

Schema-versioned, strict, and refused rather than guessed at. The reader rejects
a missing field, an **unknown** field, a boolean where a number belongs, a
non-finite value, a reason code or advice string this version does not publish,
the same reason twice, and — the shape this module exists for — a record whose
stated decision contradicts its own evidence. `ready` is derived, never stored:
it is exactly "no gate failed", so a result claiming to be ready beside failures
cannot be constructed in memory and is refused on the way back in.

The decision, the reasons, the evidence and the policy are read as **one
statement**, not four. Every per-field rule above checks one number; only the
cross-check can catch a record whose every part is individually well formed and
whose verdict is nonetheless invented. So the reader also re-derives the
required reason set from the record's own evidence under the record's own
policy and refuses any disagreement:

- an unresolved tracking loss carrying no tracking refusal — and, the other
  way, a tracking refusal beside resolved tracking evidence. A validated
  recovery is history and obliges nothing, so a record naming loss episodes
  that were closed still reads back;
- a pose, coverage or surface threshold failure missing from the reason set;
- a reason asserted while its own evidence passes;
- unmeasured evidence — coverage, surface agreement, or motion with nothing to
  measure it against — without its "not measurable" refusal beside it. "Not
  measured" is never "measured and fine";
- counts, distances, fractions and policy values outside their domains, or
  impossible beside the record's other fields: more distinct viewpoints than
  poses, a median travel above the worst one, more fusion frames sampled than
  offered, more evidence points than examined samples, a p95 past the search's
  own clamp;
- reasons in any order but the canonical one this schema publishes, because a
  front end renders the first failure's advice as the thing to do next.

A pre-fusion record whose `surface` is honestly `null` obliges nothing: the
volume does not exist, so that gate has not run and has no verdict to be
missing.

### Records no evaluation could have produced

The last rules are not about unusual scans. They are about records whose every
field is in range and which no run of this module emits, because the evaluator
has a small fixed set of *shapes* and they are not every combination of the
numbers in them:

- **The policy must describe a reachable gate.** A required share above `1`, an
  angular spread above `pi`, a zero *scale* (the noise floor, the viewpoint
  separation multiple, the comparison cell fraction — as opposed to a
  threshold, where zero is merely lax), viewpoint counting that stops below the
  number of viewpoints it requires, a coverage frame budget under the
  observations each cell must have, or both terms of the surface tolerance at
  zero. Each of those silently refuses every reconstruction forever while its
  record blames the scan. Refused where the policy is *built*, so it is refused
  in memory as well as on the way in.
- **Tracking is one of two states.** `tracking_evidence` reaches "resolved"
  only through the capture reader, which refuses any loss no validated recovery
  segment closed — so a resolved record has recovered every loss it found. It
  reaches "unresolved" only by giving up before that answer exists, so an
  unresolved or unreadable record has validated none. A resolved capture that
  found a loss and recovered nothing is a claim that the authority passed a
  capture it refuses.
- **Measured and unmeasured have shapes.** A measured surface agreement has a
  non-empty population on both sides and at least one sampled frame behind it;
  an unmeasured one reports no fraction and no distance. Unmeasured coverage
  carries no cells at all, because `measured=False` is exactly where
  `coverage_map` raised — the frame and point counts stay, since those describe
  what was offered to it. An unmeasurable trajectory reports no travel, no
  excursion, no cap and one viewpoint at most.
- **The sampling arithmetic must close.** Downsampling only shrinks, so the
  compared evidence points cannot outnumber the deprojected source points,
  which cannot outnumber the candidates examined. And every count is checked
  against the ceiling the record's *own* policy states: probe points, coverage
  points per frame times the frames used, fusion pixels per frame times the
  frames sampled, and the surface sample ceiling on either compared cloud. A
  cap that was reported as hit must equal the cap the policy states.

## Cost

Bounded, CPU-only, no new dependency, no GPU, nothing retained past the call.

- Tracking: one `stat` per run on a healthy capture.
- Poses: one pass over `poses × 128` probe points, plus greedy viewpoint
  counting against at most 16 representatives. Never all-pairs over the poses.
- Coverage: at most 64 frames by uniform stride, at most 2000 points each.
- Fusion evidence: at most 64 frames by uniform stride, at most 2048 candidate
  pixels of each by uniform stride over the raster. One frame's bytes are the
  largest thing held at a time, and a 170-frame capture costs the same as an
  8-frame one.
- Surface: both clouds voxel-thinned to one point per comparison cell; the
  search is a voxel hash over 27 cells, so it is linear in the number of
  questions asked. Measured at 0.15–0.4 s on the synthetic fixtures in the test
  module.

`scanner3d.quality.cloud_agreement` measures the same quantity and is the
repository's own instrument for it, but its search is all-pairs against a
25-million-comparison ceiling. Meeting that ceiling means thinning both clouds
until their spacing is far coarser than the truncation, and a tolerance widened
to cover that spacing can no longer see the displacement this gate exists to
catch. The hash keeps full resolution for linear work, and
`test_the_neighbourhood_search_agrees_with_the_repositorys_own_all_pairs_measure`
asserts the two agree exactly on fixtures small enough for the all-pairs version
to be affordable — so this is a cheaper route to the repository's measurement,
not a second, weaker one.

## What this deliberately does not do

No threshold was chosen to make any particular capture pass. The motion
thresholds are multiples of constants that were already calibrated for the
tracker; the surface tolerance comes from the volume's own declared numbers; the
coverage policy reuses `coverage.DEFAULT_MIN_SPREAD`. The synthetic fixtures in
the test module separate cleanly on both sides of every one of them.

It does not make tracking, registration or fusion better. It refuses results, it
does not repair them, and it cannot recover a capture that has already gone
wrong.

And it says nothing about physical accuracy. Every number here is internal
consistency. A reconstruction this module calls ready can still be the wrong
size, the wrong shape, or a beautiful model of the wrong thing.

## The capture this was calibrated against

`/home/ben/scans/scan/20260828-105933`, read-only, through
[`tests/readiness_replay.py`](../tests/readiness_replay.py) — which records
every file's size, mtime and SHA-256 before and after and reports whether they
agree.

What it establishes, and what it does not:

- **Read-only, proven.** 176 files, unchanged across every run.
- **The tracking gate concludes, and does not refuse it.** The folder holds no
  `tracking-loss.json` at all. That is not evidence the capture was healthy —
  it is the capture that *motivated* the loss record and predates it, so it has
  no durable statement of its own loss for any reader to obey.
- **The pose, coverage and surface gates cannot conclude on it.** The capture
  never reaches them: `pipeline.reconstruct_sweep` refuses frame 8 of the
  176-frame sweep outright (three independent references placed it and then
  disagreed by 6.850 against a 5.200 gate), and at stride 4 it refuses frame 1
  for a 32.441 adjacent jump against a 15.600 gate. That is the earlier
  boundary doing its job, not a limitation of these gates, and this capture's
  own defect is caught there rather than here.
