# Pose-compensated temporal depth completion

`src/scanner3d/depth_temporal.py` recovers depth samples the sensor did not
return, from the **original raw samples of other frames**, carried into the
target camera through the exact poses `scanner3d.tracking` already accepted. It
is the stage after
[confidence-aware spatial completion](depth-completion.md), and it is scored by
the same oracle, extended —
[docs/depth-completeness-benchmark.md](depth-completeness-benchmark.md).

Nothing in `scanner3d` calls it. It changes no capture, no setting, no live
preview, no tracker, no TSDF, no mesh and no export. See
[What this does not change](#what-this-does-not-change-and-why).

## The problem this exists for

The single-frame stage has an epistemic boundary, and the corpus records it
rather than describing it. In the `aperture-ambiguity` case, a one-sample gap in
a surface and a one-sample sensor dropout on that same surface present
**byte-identical** input to any method that reads one frame: the same 9×9
neighbourhood, the same single zero at the centre, the same code everywhere
else. A single-frame method fills both, and is confidently wrong about one.

No better local rule fixes that, because there is no local difference to read.
The only way out is a second measurement, taken from somewhere else. That is
what this module is: it asks what the *other* frames measured along the ray the
target frame lost, and it answers "missing" when they do not settle it.

## What it is not

- **Not a same-pixel carry.** "Use the previous frame" without applying the pose
  between the two frames is how a hole gets filled with the depth of whatever
  else happened to sit at that pixel. The corpus proves this on a scene where
  the camera really moved; see [The motion control](#the-motion-control).
- **Not a pose estimator.** Every pose it uses must already be accepted, finite
  and rigid. This module never estimates, repairs or substitutes one. An
  unaccepted pose is a refusal, not an input to be improved.
- **Not a smoother.** Raw frame bytes are immutable, temporal evidence may
  occupy only a raw target zero, and every measured code comes back
  byte-identical at confidence `1.0`.
- **Not physical evidence.** The corpus renders its own truth and hands the
  method exact poses. A real capture has neither.

## Why this is an offline post-tracking stage

Every transform here is built from a pose tracking has already accepted. So this
stage cannot run before tracking; cannot feed tracking without making tracking's
input depend on tracking's output; and cannot be run live on a frame whose pose
is not yet settled. That is a property of the method rather than a scheduling
choice. An unconfirmed pose turns a temporal reconstruction into a measurement
of the *pose error* instead of the scene — and the result would still look like
geometry.

## The coordinate and pose convention

Stated once, because a composition in the wrong order produces a scan that is
wrong in a way that still looks like a scan.

- A **camera frame** is `scanner3d.depth_metric.CAMERA_FRAME`: `+x` right, `+y`
  down, `+z` forward along the optical axis, and only `z > 0` is in front of the
  camera.
- A frame's `PoseRecord.pose` is its **world pose**: it carries that frame's
  camera coordinates into world coordinates, exactly as in `scanner3d.tracking`.
- The transform that carries a source frame's camera coordinates into the target
  camera is therefore

  ```
  temporal_transform(target, source) == target.pose.inverse().compose(source.pose)
  ```

  and composing it the other way round is invisible on a stationary pair while
  moving every sample the wrong way under motion. `temporal_transform` is public
  and its direction is asserted directly in `tests/test_depth_temporal.py`, not
  inferred from an aggregate score.

One observation therefore travels: source pixel → source camera (deprojected
with the shared intrinsics and depth scale) → world → target camera → target
pixel. A point behind either camera, or outside the target rectangle, is
discarded with its own named reason.

## The public API

| Name | What it is |
| --- | --- |
| `CameraModel` | the shared intrinsics and depth scale; deprojection and projection live here |
| `PoseRecord` | one frame's accepted world pose, its acceptance status and its quality |
| `TemporalFrame` | one frame: its index, its immutable `DepthPatch` and its `PoseRecord` |
| `TemporalPolicy` | every bound the run is held to, in codes and in pixels |
| `Observation` | one other-frame raw measurement, carried into the target, with its full provenance |
| `TemporalEvidence` | the five named facts a confidence is derived from |
| `TemporalWork` | what one run cost and every refusal it made |
| `TemporalPatch` | the whole result rectangle: codes, states, confidence, confirmations, support |
| `complete_temporally` | the entry point |
| `temporal_transform`, `viewpoint_separation`, `project_observation`, `temporal_confidence`, `select_backend`, `temporal_samples` | the pieces, public so they can be asserted on their own |

Every type is frozen, deeply copied at construction and validated there.
Direct construction refuses malformed dimensions, frame identities out of order
or repeated, raw lengths and codes that are not the frames', non-finite or
non-rigid transforms, unaccepted poses, booleans where counts belong, mutable
aliases, unsupported schema versions and unimplemented backends.

## The rules a reconstruction has to pass

1. **Only original raw measurements are evidence.** A source sample is eligible
   only when *its own* raw code is nonzero. Neither a same-frame sample, nor a
   spatial reconstruction, nor another temporal reconstruction is independent
   temporal support.
2. **Support has to be independent.** At least `min_support_frames` (default 2)
   distinct **other** frames, and at least `min_distinct_viewpoints` (default 2)
   genuinely different viewpoints among them. Two frames whose transforms into
   the target differ by less than `min_baseline` in translation *and* less than
   `min_rotation_degrees` in rotation are one viewpoint seen twice, whatever
   their indices say. A stationary camera does not become a second view by
   taking a second frame.
3. **An observation supports only the sample it is nearest to.**
   `max_reprojection_distance` defaults to 0.5 pixels. This is the single most
   consequential number in the policy and it is set conservatively on purpose:
   widening it to 0.6 on the three-view synthetic case raises recovered samples
   from 16 to 195 and the worst error against held-out truth from 1 raw code to
   205, because an observation two thirds of a pixel away is a measurement of a
   *nearby* point on the surface, and on a surface turning away from the camera
   those are not the same depth.
4. **Nearer measured geometry wins.** Every observation is deposited before any
   of them is judged, and a z-buffer rule then discards evidence hidden behind
   nearer evidence by more than `occlusion_margin_codes`. Deciding occlusion
   while depositing would let iteration order pick the answer.
5. **Disagreement is detected before it is averaged.** Per-frame representatives
   must agree within `max_depth_disagreement_codes`; a wider spread means the
   frames are looking at different surfaces and the sample stays missing.
6. **Discontinuities are refused.** A hole whose own measured neighbours straddle
   a step wider than `cluster_gap_codes` is the silhouette, the disocclusion and
   the grazing edge — where "which surface is this" is exactly the question the
   evidence cannot settle.
7. **Work is bounded before the loops.** `TemporalPolicy.worst_case_work` is
   exact and derivable from the input before the run:
   `eligible_source_samples × footprint_cells + holes × (2 + max_candidates_per_target)`.
   Each eligible source sample is projected once and deposited into — and later
   judged for visibility at — at most `footprint_cells` target samples. Each
   hole then costs one unit for being examined plus one per candidate deposited
   at it, and that is `max_candidates_per_target + 1` rather than
   `max_candidates_per_target`: the deposit deliberately keeps one observation
   past the cap so an over-cap target is *provably* over it rather than
   silently truncated to it, and that retained sentinel is processed like any
   other candidate. Work a run does is work its budget covers.
   A call past `MAX_WORK_UNITS` (25,000,000) raises `WorkTooLarge` before the
   first projection. Frames, support frames, footprint radius, candidates per
   target and patch samples all have absolute product ceilings too.

   One consequence worth stating, because it is easy to assert the opposite:
   the number of deposited candidates may legitimately **exceed** the number of
   projected observations. A projection landing on an exact half-pixel boundary
   is at `max_reprojection_distance` from both adjacent samples and supports
   both. The only honest bound between the two counts is
   `candidates ≤ observations × footprint_cells`, and that is what the result
   type and the benchmark's strict reader check.

Every hole that is not reconstructed is counted against one of a closed
vocabulary of reasons, always fully reported at zero when it never fired:
`no_posed_evidence`, `all_evidence_refused`, `too_many_candidates`,
`insufficient_support_frames`, `insufficient_viewpoint_diversity`,
`conflicting_depth_evidence`, `edge_discontinuity`, `prediction_out_of_range`,
`confidence_below_threshold`. Individual observations are discarded against a
second closed vocabulary: `source_behind_camera`, `behind_target_camera`,
`outside_target_patch`, `reprojection_too_far`, `target_already_measured`,
`predicted_code_out_of_range`, `occluded_by_nearer_evidence`.

## Spatial confirmation is not temporal recovery

An accepted `CompletedPatch` for the target frame may be handed in. Where it
proposes a value at a target hole, this module publishes what the independent
posed evidence said about it, for **every** sample, in a closed vocabulary.

| Verdict | What happened | Provenance of the sample |
| --- | --- | --- |
| `confirmed` | independent posed measurements reproduced the estimate within `confirmation_tolerance_codes` | `temporally_reconstructed`, carrying the temporal value, naming both the other-frame raw support and the spatial support it confirms |
| `contradicted` | independent posed measurements produced a value the estimate does not match | `temporally_reconstructed`, carrying the **independent** value; the overruled estimate's own support stays named |
| `not_observed` | no independent evidence reached the sample | `spatially_reconstructed`, with the spatial code and the spatial confidence unchanged |
| `no_spatial_estimate` | nothing was proposed there | temporally reconstructed if the posed evidence suffices, missing otherwise |

Three things about that table are load-bearing.

**A contradiction is never averaged.** A number halfway between two incompatible
answers is evidence of neither. The independent value is published whole and the
disagreement is published beside it.

**A contradiction overrules rather than defers.** The evidence at a contradicted
sample is exactly the evidence a run given no spatial estimate has there, and
such a run reconstructs it. Letting a weaker same-frame guess veto a
well-supported independent measurement would make the published depth depend on
whether an earlier stage happened to guess — which is not a property of the
evidence. `TemporalPatch` enforces this: a sample labelled `contradicted` whose
published code is *within* the confirmation tolerance of the spatial estimate is
refused.

**A contradiction is not a recovered sample.** It is separate evidence about the
*estimate*. The recovered sample is the independent measurement that replaced
it, and that is the only thing the benchmark's coverage gain may be read from —
see `coverage.temporally_reconstructed_on_surface` in the benchmark document.

An estimate nothing independent reached keeps its spatial provenance explicitly.
An unsupported spatial hypothesis is not made safe by being run through a
temporal stage, which is the specific failure this vocabulary exists to prevent.

## Confidence

`TemporalPatch.confidence` is a fraction of 1 derived from named evidence, never
a constant. It is the geometric mean of five terms, so that every one of them
can veto and none can be traded away:

| Term | Formula | Direction |
| --- | --- | --- |
| `support` | `(frames - min_support_frames + 1) / (max_support_frames - min_support_frames + 1)` | more distinct supporting frames raises it |
| `diversity` | `viewpoints / max_support_frames` | more genuinely different viewpoints raises it |
| `proximity` | `1 - mean_distance / (max_reprojection_distance + 1)` | landing further from the sample lowers it |
| `agreement` | `1 - disagreement_codes / max_depth_disagreement_codes` | disagreeing about the depth lowers it |
| `pose` | worst `PoseRecord.quality` among the supporting frames | a reconstruction is never worth more than the pose that placed its evidence |

Each term is clamped to `[0, 1]` and is monotone in the direction its name
implies, so the mean is too — which is what a caller thresholding on confidence
depends on, and it is tested directly. The result is capped at
`MAX_RECONSTRUCTION_CONFIDENCE` (0.99), strictly below the `1.0` a measured
sample carries, so the two populations cannot meet even when every term is at
its best. It is a **reconstruction** confidence: it says nothing about the
sensor, and no value of it ever makes a reconstructed sample a measurement.

## A result is re-derived, not believed

A result whose every label and index is legal can still carry a reconstructed
code, a confidence, a support selection, a confirmation or a work total that the
evidence does not give. So `TemporalPatch` carries the exact frames, camera,
accepted spatial result and policy it was derived from as **fields**, and its
constructor re-derives the whole run from them and requires the result to *be*
that derivation.

The derivation is a pure function of values the result cannot change, so a
forged field is refused by recomputation rather than by consistency with its
neighbours. This is what catches the one forgery the shape rules cannot: a
sample the derivation *contradicted*, relabelled `confirmed`. Both verdicts leave
the sample `temporally_reconstructed`, so no count and no state disagrees — only
recomputation does.

## What the corpus says about it

Run the oracle with:

```bash
.venv/bin/python -m tests.depth_completeness_harness --out depth-completeness-report.json
```

Two cases were added for this stage, and the accepted `multiview-object` case
gained the recovery and temporal-work gates and the temporal candidates.

### The motion control

`temporal-motion` is one box in front of a backing plane, from three laterally
displaced cameras, each missing a complementary strip of columns. The camera
moves exactly five pixels at the box's own depth between frames, so the
silhouette sits at column 26 in frame 0, 31 in frame 1 and 36 in frame 2, and
frame 0's strip straddles its own silhouette.

Carrying the other frames' samples at the **same pixel** therefore lands backing
plane depth — 17,920 raw codes — on samples whose truth is the box front face at
12,288. That is a 5,632-code error on the wrong surface, and it is not an
aggregate score: the two answers differ by which surface a sample is on. The
`naive_temporal_carry` control is the **densest** result in the case and fails
the error, edge, step-column and confidence gates. Carrying the same raw samples
through the exact poses puts them where they belong.

The product's `temporal_completion` passes every error, edge, step-column,
dimension, confidence, work and recovery gate on that case and fails only
`coverage.supported_fraction`, because the speckle holes that are in every frame
are in no other view either. The scene owes 265 samples away from its declared
foreground boundary; the run recovers 284, refusing one at the silhouette under
rule 6 above.

### The confirmation control

`temporal-confirmation` contains one genuinely empty aperture, one held-out
plane sample and one held-out sample of a one-sample-wide pillar standing in
front of the plane, from three posed views. From inside one frame all three look
identical. Independent posed evidence reaches all three verdicts: the plane
sample is confirmed, the pillar sample is contradicted and republished as the
independent measurement, and the aperture is never observed — no sample of any
other frame lies on its ray.

So `spatial_then_temporal_completion` still fails `coverage.false_fill` and
`confidence.high_confidence_false_fill` on that case. The accepted single-frame
ambiguity is **disclosed** by the second stage rather than laundered by it, and
that failure is declared in the fixture.

## What this does not change, and why

Nothing in `scanner3d` calls this module. No capture path, no scanner setting,
no GUI, no tracker input, no TSDF, no mesh, no export and no live preview reads
a temporal reconstruction. Tracking continues to consume measured geometry only.

That boundary has a measurement behind it rather than only an argument. On
`multiview-object`, `temporal_completion` recovers exactly the 16 samples
another posed view really measured, every one of them within a single raw code
of the truth, and invents nothing. Its **gated** tracking evidence —
`tracking` and `tracking_pose`, the tracker over measured samples only, which
is the route the product takes — is the baseline's exactly, acceptance and
poses alike. Beside it the benchmark publishes an ungated counterfactual,
`tracking_completed_cloud_pose`: the same tracker over the completed cloud. It
says that feeding those 16 accurate reconstructions to the real tracker moves
where it places frame 2 from 1.09 degrees to 2.38, past the bar the baseline
sets. That record carries no gate, because a bar over a route this project does
not take would read as permission to take it; it is published because it is the
measured reason the route stays closed. ICP over a cloud that sparse is
sensitive to which points exist, and being right about a sample is not the same
as being safe to feed to a stage that was fitted without it.

Whether a temporal reconstruction is eligible for fusion, and at what weight, is
a decision a later stage makes with its own evidence. This module exists so that
decision has something honest to make it from. That stage is now defined —
[docs/weighted-tsdf-input.md](weighted-tsdf-input.md) — and it reads this
module's accepted result rather than a caller's labels: an accepted temporal
reconstruction is offered to fusion at `0.5 x` its confidence here, an
unconfirmed spatial estimate at exactly zero. It changes no export: the
reconstruction export still fuses its original raw frames.

Finally: the poses in every case above are the scene's own, exact. A real
capture does not have them for free, the truth is rendered by
`scanner3d.simulate` and is not a sensor model, and no number here is a physical
measurement of any scanner.
