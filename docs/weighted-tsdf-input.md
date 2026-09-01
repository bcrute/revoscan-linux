# Provenance-weighted TSDF input

`src/scanner3d/tsdf_input.py` defines what one depth frame is allowed to be
*worth* to `scanner3d.tsdf`, and
`scanner3d.tsdf.TsdfVolume.integrate_weighted` is the entry point that spends
it. Together they are the fusion **input contract**: a frame whose every sample
carries the provenance it came from and a finite fusion weight derived from
that provenance, and nothing else.

Nothing in `scanner3d` calls either of them. The reconstruction export still
fuses its original raw frames through the unchanged `integrate`, and no
capture, setting, GUI, live preview, tracker, mesh or export path reads a
weighted frame. See
[What this does not change](#what-this-does-not-change-and-why).

**The measured outcome is negative, and it is stated up front.** On the
synthetic corpus, fusing accepted temporal reconstructions through this
contract improves no coverage or surface measure at all against the same raw
multi-view frames, because what those reconstructions recover is what the
supporting frames measured and the baseline fuses those frames too. The
contract itself works and is measured; the use case does not appear. See
[the negative result](#what-the-synthetic-corpus-measured).

## The problem this exists for

`TsdfVolume.integrate` takes raw frames, and every sample the sensor returned
weighs exactly the same as every other. That is the honest default while the
only thing a frame carries is measurement: the running weighted mean a TSDF
keeps is the minimum-variance estimator precisely when the observations are
equally uncertain.

[Spatial completion](depth-completion.md) and
[pose-compensated temporal completion](depth-temporal.md) now produce something
a raw frame cannot express — a sample the sensor did **not** return, standing
where independent posed views agree a surface is, carrying an exact confidence
derived from that evidence. Handing such a sample to `integrate` would fuse it
as a measurement, which is the one thing this repository has spent three tasks
refusing to do. Dropping it loses evidence that other views really did measure.

This module is the third option, and only the third option. It converts
nothing, estimates nothing and completes nothing: the two accepted completion
modules own every derivation, and this one is refused the right to duplicate
them.

## The weighting contract

Four provenances, and exactly four. Each weight is fixed in the module and is
**not a caller's parameter**, because a weight a caller could raise is not
evidence about anything.

| provenance | offered code | fusion weight |
| --- | --- | --- |
| `measured` | the raw code, byte for byte | `MEASURED_WEIGHT` = `1.0` |
| `temporally_reconstructed` | the accepted `TemporalPatch` code | `TEMPORAL_WEIGHT_CEILING * confidence` |
| `spatially_excluded` | `NO_RETURN_SAMPLE` | `0.0` |
| `missing` | `NO_RETURN_SAMPLE` | `0.0` |

### The exact formula, and why it is that one

```
temporal_fusion_weight(confidence) = TEMPORAL_WEIGHT_CEILING * confidence
                                   = 0.5 * confidence
```

`confidence` is `scanner3d.depth_temporal`'s own accepted value, republished
exactly and never re-derived here. That module accepts nothing below its
policy's `min_confidence` (`0.2` by default) and publishes nothing above
`MAX_RECONSTRUCTION_CONFIDENCE` (`0.99`), so an accepted reconstruction weighs
between **0.1 and 0.495**.

One multiplication, and deliberately nothing more elaborate. Three properties
are wanted and this is the simplest function that has all three:

- **strictly increasing** in the confidence, so lowering an accepted
  reconstruction's confidence can only lower its influence on the field;
- **strictly positive** at every confidence that module can accept, so an
  accepted reconstruction is never silently discarded;
- **strictly below** `MEASURED_WEIGHT` at every one of those confidences, with
  the widest margin at the top rather than the narrowest.

The ceiling is half a measurement, and that is not tuning. At `0.5`, two
agreeing reconstructions still weigh less than the one measurement they
disagree with, so a voxel a camera actually saw is never carried away from what
it saw by reconstructions alone. `tests/test_tsdf.py` asserts that directly.

Because the weight is one multiplication of one float, two runs on the same
confidence produce the identical float on either backend.

### Why an unconfirmed spatial estimate weighs nothing

Not a low weight — **zero**, which integration treats as absence.

[Task 029's aperture-ambiguity case](depth-completion.md) is the reason: a
one-sample gap in a surface and a one-sample dropout *on* that surface present
byte-identical neighbourhoods to any method reading one frame. A spatial
estimate under that ambiguity cannot be told from invented geometry from inside
the frame that made it, and a hypothesis that can add weight to a voxel is
invented geometry.

Such a sample becomes eligible for weight only when an independent view settles
it. `TemporalPatch.confirmation_at` says `confirmed` or `contradicted` only
where independent evidence did, and in **both** of those cases the accepted
result has already republished the sample with the independent value,
confidence and support — which is what is weighted. Where it says
`not_observed`, the sample is still the single-frame hypothesis and still
weighs nothing.

A `missing` sample — nothing returned it and nothing reconstructed it — is
what a raw no-return already was, and weighs nothing for the same reason.

## The API

### Building a weighted frame

Three thin adapters, one per accepted producer. Each takes the exact 512000
raw bytes and, where there is one, the **already validated** completion result
those bytes were completed into:

```python
from scanner3d import tsdf_input

raw_frame = tsdf_input.weighted_raw_frame(raw)
spatial = tsdf_input.weighted_completed_frame(raw, completed, origin=(u0, v0))
temporal = tsdf_input.weighted_temporal_frame(raw, temporal_result, origin=(u0, v0))
```

- `weighted_raw_frame` is the **identity adapter**: every return at full weight,
  every zero at none. A frame built here fuses byte-for-byte identically to
  handing the same bytes to `integrate`.
- `weighted_completed_frame` adds nothing to the field, deliberately. Every
  spatial reconstruction arrives `spatially_excluded` with a no-return code, so
  the field is the raw frame's field and the *count* of what was excluded is
  published rather than lost.
- `weighted_temporal_frame` is the only adapter that can add weight to a voxel
  the sensor did not reach.

`origin` places the completed rectangle inside the 640×400 frame; a rectangle
that would leave the frame is refused.

### `WeightedDepthFrame`

Deeply immutable. Every buffer is copied on the way in, the sparse weight
mapping is re-encoded into immutable `bytes` that refuse rebinding, and the
accepted completion result it binds to
is itself a frozen, self-validating type — so nothing a caller still holds can
change what a frame means after it has been checked.

| field | what it is |
| --- | --- |
| `raw` | the exact 512000 bytes, and the authority for every measured sample |
| `codes` | the codes offered to fusion |
| `provenance` | one byte per sample, from `PROVENANCE_NAMES` |
| `weights` | `SparseWeights`: `{sample index: fusion weight}`, temporal reconstructions only |
| `summary` | the exact counts and total weight (see below) |
| `adapter` | which accepted producer built it |
| `completion` | the accepted `CompletedPatch` / `TemporalPatch`, or `None` |
| `origin` | where the completed rectangle sits, as `(u0, v0)` |

It is validated by **re-derivation** rather than field by field: the whole frame
that `raw`, `completion` and `origin` imply is rebuilt in `__post_init__`, and
the offered codes, provenance, weights and summary are required to *be* that
derivation exactly. So a forged frame has nothing to forge with — relabelling a
sample `measured` contradicts `raw`, and relabelling one
`temporally_reconstructed` contradicts `completion`, which has already bound
itself to its own raw inputs and re-derived its own run.

Refused, each before the frame exists, so a refusal leaves no half-built
provenance behind it: a buffer that is not one whole frame; a provenance byte,
adapter or schema version nobody defined; a boolean where a number belongs; a
non-finite or out-of-range weight; any weight at all on a missing or excluded
sample; a zeroed weight on an accepted reconstruction; a changed measured value
or weight; a reconstruction without an accepted source; a result derived from
other bytes; and a summary whose counts are not the counts the frame carries.

### `WeightedFrameSummary`

A partition: `measured + temporally_reconstructed + spatially_excluded +
missing` equals the frame's sample count, which is the denominator the capture
mode fixes. `total_weight` is `math.fsum` over every sample's fusion weight, so
it is the correctly rounded sum rather than an order-dependent accumulation.

### Weighted integration

```python
volume.integrate_weighted(frames, poses, intrinsics=..., depth_scale=...)
```

The same projective arithmetic `integrate` does, with one change and only one:
the geometric weight the configured `WeightPolicy` gives an accepted voxel
observation is multiplied by the sample's own fusion weight. Poses run in the
same direction `integrate` documents — each maps *its frame's* camera
coordinates into the model frame.

A call is **validated whole, before the field is touched**. A frame that is not
a `WeightedDepthFrame`, a pose count that does not match, more frames than the
per-call ceiling, or unusable intrinsics or depth scale leaves every value,
every weight, every evidence count and every frame total exactly as it found
them, and increments only the refusal counter — so a run that was offered
something malformed does not read like a run that was not.

## How it interacts with the existing weight policies and the cap

The fusion weight **multiplies** the geometric weight; it does not replace it.

- Under `WeightPolicy.UNIFORM`, an accepted observation contributes exactly the
  sample's fusion weight.
- Under `WeightPolicy.RAMPED_BEHIND`, an observation in the truncation band
  behind the surface already contributes `1 + normalised` (falling from 1 at
  the surface to 0 at the truncation). A reconstruction there contributes
  `fusion * (1 + normalised)`. The two effects compose rather than either one
  overriding the other, which `tests/test_tsdf.py` asserts against the product
  of the two factors measured separately.

`TsdfConfig.max_weight` still caps a voxel's *accumulated* weight, and the cap
is reached by accumulating weight rather than by counting observations. So six
measured frames reach a cap that six reconstructions of the same geometry do
not — a lower weight keeps meaning a lower weight all the way to the ceiling.

A zero-weight sample takes the **same path a raw no-return takes**: it updates
no voxel, it is counted under `IgnoredReason.NO_RETURN` in the base accounting
and under `zero_weight_observations` in the weighted record, and it cannot turn
an unobserved voxel into an observed one. Absence and zero weight are the same
thing, deliberately, and the corpus proves the fields are byte-identical.

## What the run publishes

`TsdfVolume.weighted_evidence()` returns an immutable
`WeightedIntegrationEvidence` **beside** the base `evidence()` rather than
instead of it: the base record still partitions every `(frame, voxel)` pair the
volume was offered, and this one says what the provenance and the weights added
on top.

- `inputs` — the four provenance counts, which partition
  `frames_offered × 256000` samples, and `offered_weight`, the frames' own
  correctly rounded totals summed.
- `voxel_updates`, `zero_weight_observations`, `refusals`.
- `contribution` — the total fusion weight that actually reached voxels: the
  sum, over accepted voxel observations, of the geometric weight times the
  sample's fusion weight.
- `contribution_by_provenance` — the same total split four ways.

Every number here is **accounting**: it says how much evidence reached the
field and under which provenance. None of it is a geometric measure, and none
of it may stand in for one — see the negative result below, where 225 extra
voxel updates buy exactly zero observed voxels, surface vertices and coverage.

### On exactness

`contribution` is accumulated as a compensated (Neumaier) sum over per-chunk
`math.fsum` partials taken in flat-voxel-index order, which is a fixed order
both backends visit voxels in. It is therefore **deterministic, and identical
on the reference and the accelerated path** rather than merely close. It is
**not** claimed to be the exactly rounded sum of every individual product, and
the record says so. `contribution_by_provenance` is accumulated exactly the
same way, from the same per-chunk partials, so the split has the same property.

The split exists so the contract's central zero is a **measurement** rather
than an assertion. Nothing in the record states that an unconfirmed
single-frame estimate contributed nothing; the accumulator does, because a
weighting defect that let one through would land in that entry and nowhere
else — and the benchmark gates it at exactly `0.0`.

## Work and storage bounds

Stated so a reader can check them rather than take the word "bounded".

- One weighted frame owns `FIXED_FRAME_BYTES` = **1 280 000 bytes** — the raw
  frame it preserves, the codes it offers, and one provenance byte per sample —
  plus `RECONSTRUCTION_ENTRY_BYTES` = **12 bytes** per temporally reconstructed
  sample. That is the whole cost: `SparseWeights` holds the reconstructions as
  two `bytes` objects, a little-endian `uint32` index beside a little-endian
  `float64` weight, and `WeightedDepthFrame.owned_bytes` is the length of the
  four buffers added up rather than a constant that ignores the sparse part.
- At the worst frame this contract allows — every one of the 256 000 samples a
  temporal reconstruction — that is `MAX_FRAME_BYTES` = **4 352 000 bytes**,
  and `tests/test_tsdf_input.py` demonstrates the bound at exactly that
  population rather than arguing for it at a typical one. `sys.getsizeof` is
  deliberately not part of the contract: it measures one interpreter's
  container overhead and is not portable. A `dict` of 256 000 boxed integer
  keys and boxed float values, which is what this used to be, has no portable
  byte count at all, which is why the encoding changed.
- A long offline sweep therefore costs frames times a constant, never a growing
  per-frame structure. The compactness costs no precision: `float64` is the
  width a Python float already is, so a weight read back out is the identical
  float the accepted confidence produced, on both backends.
- Building a frame is one C-level pass over the raw bytes (`bytes.translate`
  over the two ORed byte planes) plus one interpreted pass over the completed
  rectangle, which is bounded by the rectangle the caller fixed.
- Validating a frame is the same derivation again, plus four `bytes.count`
  passes and one `math.fsum` for the summary.
- Weighted integration is the unchanged per-voxel cost of `integrate` plus one
  byte index per voxel on the reference path, and a binary search over the
  index buffer only at the reconstructions themselves; the
  accelerated path builds one dense per-sample weight array once per frame
  (2 MiB, a fixed cost beside the volume) and the per-chunk arithmetic stays a
  single indexed multiply.
- `MAX_FRAMES_PER_CALL` still bounds one call, and the volume's memory does not
  grow with frame count.

## What the synthetic corpus measured

All of it offline, on rendered scenes with exact poses, through
[the depth-completeness benchmark](depth-completeness-benchmark.md). Two of its
ten cases fuse a weighted volume, and they carry it for opposite reasons.

**`multiview-object` — the negative result.** Three frames of complementary
loss, one volume, one configuration, compared against **the same raw multi-view
field**: the same three frames, the same poses, the same volume, the same
extractor, the same truth model.

| | raw multi-view baseline | `temporal_completion` | difference |
| --- | ---: | ---: | ---: |
| temporal reconstructions offered | 0 | 16 | +16 |
| offered weight | 487.0 | 492.322559… | +5.322559… |
| voxel updates | 5728 | 5953 | +225 |
| contribution | 5728.0 | 5802.848236… | +74.848236… |
| **observed voxels** | 3099 | 3099 | **0** |
| **surface vertices** | 333 | 333 | **0** |
| **truth-referenced surface coverage** | 0.9647630619684082 | 0.9647630619684082 | **0.0** |

The first four rows are **accounting**: they say that evidence reached the
field. The three bold rows are the **output**, and the gain there is exactly
zero. There is no geometric improvement to claim, and none is claimed.
The corpus publishes `observed_voxel_gain`, `surface_vertex_gain` and
`surface_coverage_gain` — this field's output measures minus that baseline's —
and gates all three for non-inferiority only. The gain gate that stood over
`voxel_updates` in an earlier round has been removed: an internal update count
is not a coverage or surface measure, and gating it would have dressed
redundancy up as an improvement.

**Why it is zero, and it is not the fixture's fault.** An accepted temporal
reconstruction is a depth another posed frame *measured*, carried into this
frame through the pose between them. The 3D point it implies is therefore a
point one of the support frames already put in the volume. Once those support
frames are fused — and they must be, they are part of the same scan — the voxel
is already observed, and all the reconstruction can do is add weight to an
observation that is already there. That is what the +225 voxel updates are.

The corpus proves this rather than asserting it, by computing the comparison
that *would* have shown a gain and naming it for what it is. Fuse the single
frame carrying the most reconstructions **on its own**, with its support frames
withheld:

| | raw | weighted |
| --- | ---: | ---: |
| observed voxels, one frame alone | 1949 | 2174 |

Those +225 voxels are the same 225 updates. The reconstructions are real
geometry that a lone frame does not have — and every bit of it is already in
the multi-view baseline, because it came out of the very frames that baseline
fuses. `withheld_support_observed_voxels` and
`withheld_support_weighted_observed_voxels` are published under those names,
carry no gate, and are not evidence of an improvement. The strict reader binds
`measured_only_frames` to every frame the record was offered, so a report
cannot quietly put the withheld number forward as its baseline.

**What follows for the roadmap.** On this evidence, routing accepted completed
samples into the reconstruction export would add weighted observations and no
geometry, at the cost of a provenance-carrying path through fusion. The
recommendation from this task is that the export continues to fuse **raw
multi-view frames** and that the later export-routing stage be dropped unless a
scene is found where the support frames genuinely cannot all be fused — a
single-view or severely truncated sweep, which this corpus does not contain and
which is not what the scanner does. That decision belongs to the planner.

The 16 reconstructions offer 5.32 of weight between them, an average of 0.333
each, which is below `MEASURED_WEIGHT` and above zero exactly as the formula
requires. The weighting contract itself is intact and measured; what is absent
is a use for it.

**`aperture-ambiguity` — evidence refused.** One frame in which the accepted
single-frame primitive fills both a real gap and a dropout it cannot tell apart:

- 2 samples arrive `spatially_excluded`, and `spatially_excluded_contribution`
  is `0.0` — measured, and gated at exactly zero;
- `matches_measured_only` is true: the weighted field is **byte-identical**,
  values and accumulated weights both, to the field this scene's measurements
  alone produce;
- the counterfactual beside it — the same candidate's completed codes fused at
  uniform weight, which is what ignoring provenance would do — observes 8457
  voxels against 8445 and pulls out 897 surface vertices against 893. Those
  extra voxels are the aperture being filled, and they are exactly what the
  weighting refuses.

That counterfactual is why the exclusion is load-bearing rather than
decorative, and it is kept in the corpus for that reason.

A candidate that carries **no accepted product result** reaches no weighted
volume at all. `dense_nearest` fills the aperture at confidence 1.0 and could
just as easily have filled it at 0.1; it has no accepted result to offer, so it
has no weight to be low. That is the difference between provenance and a label,
recorded as the absence of the only route rather than as a gate it happens to
fail.

## What this does not change, and why

Nothing in `scanner3d` calls `tsdf_input` or `integrate_weighted`. The GUI, the
live preview, the capture path, the tracker and `scanner3d.reconstruction_export`
are unchanged, and the export still fuses its original raw frames through
`integrate`. Measured-only tracking remains exactly the baseline; nothing
completed enters tracking.

The unchanged raw API is byte-for-byte what it was. A raw frame through
`weighted_raw_frame` and `integrate_weighted` produces the identical stored
values, the identical accumulated weights and the identical base evidence as the
same bytes through `integrate`, on both backends and under both weight
policies — the equivalence is not approximate, it is multiplication by one.

Routing an accepted weighted frame into an export is a later, separately
reviewed task.

## Limitations

- **Offline synthetic evidence only.** Every number above comes from rendered
  scenes with exact poses. A real capture has neither, `scanner3d.simulate` is
  not a sensor model, and **no physical improvement is claimed** for any
  scanner, including the attached MINI.
- A fusion weight is evidence about **provenance**, not about accuracy. A
  reconstruction at high confidence is still a reconstruction; the field records
  how much it was allowed to count for, not that it was right.
- The ceiling and the linear form are a stated contract, not an estimate of a
  reconstruction's variance. Nothing here has measured what a temporal
  reconstruction's uncertainty actually is on real data.
- No dimensional accuracy is claimed by any record this module or
  `scanner3d.tsdf` publishes.
