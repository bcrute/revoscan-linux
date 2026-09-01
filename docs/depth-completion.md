# Confidence-aware spatial depth completion

`src/scanner3d/depth_completion.py` recovers depth samples the sensor did not
return, from evidence inside the **same raw frame**, and refuses the ones that
frame does not settle. It is the first real completion stage in this repository
and it is scored by the checkpoint-1 oracle that was written before it —
[docs/depth-completeness-benchmark.md](depth-completeness-benchmark.md).

Nothing in `scanner3d` calls it yet. It changes no capture, no live preview, no
tracker, no TSDF, no mesh and no export. See
[What this does not change](#what-this-does-not-change-and-why) for why that is
deliberate rather than unfinished.

## The problem, stated precisely

A raw depth code of zero is **no measurement** — that is
`scanner3d.depth_metric.NO_RETURN_SAMPLE`'s rule and every part of this module
is built on it. Holes are therefore identifiable, a fill can only ever occupy
one, and a reconstructed sample can always be told from a returned one.

Some holes are one sample the sensor missed with measurements all around it.
Some are the shadow of a depth edge, the outside of a silhouette, or a real gap
between two objects — and there, the surface a local method would invent is
exactly the thing nothing measured. **The whole design problem is telling those
apart from one frame,** and the answer to the second kind has to be "missing"
rather than a plausible number at a low confidence.

## The public API

| Type | What it is |
| --- | --- |
| `DepthPatch` | one immutable rectangle of raw uint16 codes; `len(codes)` is always `width * height`, so every count has a denominator the caller fixed in advance. `DepthPatch.from_frame` reads one out of a whole little-endian depth frame |
| `CompletionPolicy` | every bound the method is held to, validated on construction and published in the result |
| `CompletedPatch` | the same rectangle with a state, a confidence and a support list per sample, plus the work record **and the immutable `DepthPatch` it was derived from** |
| `CompletionWork` | what the run cost and what it refused, by reason |
| `complete_patch(patch, *, policy, backend)` | the whole operation. Pure, deterministic, bounded |
| `support_confidence(fit, policy)` | the confidence formula, separately callable and separately tested |
| `hole_components(patch)`, `fit_support(support)` | the two pieces the method is built from |
| `reconstructed_samples(result)` | every reconstruction as `(index, code, confidence)`, in index order |

`CompletedPatch` stores provenance as one byte per sample (`state_codes`) and
reads it back through `state_at`, so a full 640x400 frame's provenance is half a
megabyte rather than a tuple of a quarter of a million strings. Codes,
confidence and support are the whole rectangle, every time; a result that is not
the whole rectangle is refused (`ResultUnusable`), because a result that dropped
its hardest samples cannot be counted against the denominator it was given.

### A result is bound to the frame it claims to preserve

`CompletedPatch.source` is the exact immutable `DepthPatch` the result was
derived from, and it is a **field**, not a construction-time argument that is
thrown away. Everything the result claims is validated against `source.codes`
rather than against the result's own per-sample labels, because a label is
precisely what a forged result would change: calling a sample `measured` says
nothing about whether the sensor returned it, and believing a support index
because the result labels that index measured is letting a claim vouch for its
own evidence.

With the source in hand there is exactly one shape a valid result can have, and
the constructor refuses everything else:

| For each source sample | The result must be |
| --- | --- |
| a returned code (nonzero) | the same code byte-for-byte, state `measured`, confidence exactly `1.0`, no support |
| a no-return code (zero) | either exactly `(code 0, confidence 0.0, no support, missing)`, or a `spatially_reconstructed` sample with a nonzero code, a finite confidence strictly between `0.0` and `1.0`, and distinct support indices **whose source codes are nonzero** |

The work record is bound the same way: `holes` and `attempted` are the source's
own no-return count, `accepted` and `rejected` are what the result actually
carries, `support_samples` is what it actually names, and `work_budget` is
`policy.worst_case_work(holes)` exactly — not a number the run chose. Its
`backend`, `policy` type, `schema_version`, and every count's type and range are
validated too; a boolean is not a count.

Deep immutability is a property rather than a word. Every sequence and mapping
handed to `DepthPatch`, `CompletedPatch` or `CompletionWork` is copied at
construction and republished read-only (`support` and `rejections` come back as
mapping proxies), so mutating the list, bytearray or dict a caller still holds
cannot change a constructed value — and a value that could change after
validation was never validated.

### And a result is the derivation that source and policy give

Shape is not substance. A forged result can satisfy every rule in the table
above and still publish a reconstructed code of `65000` where its own evidence
gives `100`, a confidence of `0.99` where the formula gives `0.31`, a support
list that is a *different legal selection* from the one the policy makes, a
`work_units` of `0`, or a rejection taxonomy that moved all 81 real refusals
from the reason that fired onto one that never did. Every one of those keeps
the real source, the real labels and real measured support indices, and every
one of them was accepted before round three.

So `CompletedPatch` re-derives the entire run from `source` and
`work.policy` — the two immutable things it names — and requires the result to
**be** that derivation:

| Bound by recomputation | Refuses |
| --- | --- |
| each sample's state | a hole this policy refuses published as a recovery, and a recovery published as refused |
| each reconstructed code | any value the named evidence does not give |
| each confidence | any number other than the one `support_confidence` computes |
| each support list | changed, omitted **and** extra support, even when every index in it is a real measurement |
| `work_units` | a spend the run did not make, in either direction, under the budget or not |
| every `rejections[reason]` | a taxonomy that only adds up to the right total |

The derivation reads the source and the policy and nothing else, so no part of
a result is ever evidence for another part of it, and a forged field is refused
by recomputation rather than by agreement with its neighbours. It is memoised
on those two frozen values, so a completion still costs one pass rather than
two.

**What this does not establish**, stated rather than implied: a policy edit that
changes nothing this frame is sensitive to leaves an identical derivation and is
accepted. Every published code, confidence and support selection still
reproduces exactly under the policy the result carries — which is what a reader
of the result needs — but "this is the policy the caller passed" is not a fact a
single result contains, and this module does not claim it.

Numbers whose identity is load-bearing are type-checked before they are
compared, because `True == 1` and `1.0 == 1`: a result's `width`, `height` and
`schema_version`, and `CompletionWork.schema_version`, are refused when they are
booleans or floats that merely equal the right answer.

Three states, and no temporal one: this module reconstructs only from the frame
it was handed, so a state it cannot produce is a state it does not define.

| State | Means |
| --- | --- |
| `measured` | the sensor returned this sample; the code is byte-for-byte the raw one, and the confidence is `1.0` |
| `spatially_reconstructed` | recovered from named original measurements **in this frame**; confidence is strictly below `1.0` |
| `missing` | not returned and not recovered: raw code zero, confidence zero, and a counted reason in the work record |

## The method

Deterministic, single-pass, and bounded by construction.

1. **Holes are grouped, not just counted.** Four-connected components of raw
   zeros. A component with more than `max_component_samples` samples, or longer
   than `max_component_span` in either direction, is refused **whole**. This is
   what keeps a silhouette's background, a band of dropout hugging a depth step,
   a collapsed column and the gap between two objects out: in each of those the
   surface a local method would invent spans further than any local evidence
   reaches.
2. **Support is original measurement only.** The nearest `max_support` measured
   samples inside a disc of `radius`, ordered by distance and then by position
   so the choice is total and stable. Support is read from the input patch,
   which is immutable, and never from the result being built — so no
   reconstruction is ever evidence for another one and a fill cannot creep
   across a frame one sample at a time.
3. **The support has to surround the sample.** At least `min_support` of them,
   with measured samples on both sides along both axes, and a second-moment
   spread of at least `min_spread` in its weaker direction, taken **about the
   hole** rather than about the support's own centre. Interpolation inside
   evidence, never extrapolation past its edge, and never along a single line of
   samples that fixes no surface at all.
4. **Incompatible depth is detected before it is averaged.** The support's codes
   are sorted; a gap between consecutive values wider than `cluster_gap_codes`
   means the neighbourhood holds two surfaces, and the sample is refused rather
   than placed in the empty band between them. A total spread wider than
   `max_support_span_codes` is refused for the same reason at a coarser scale.
5. **The fit has to actually fit.** A least-squares plane in
   `(column, row, code)` over the support, refused if its worst residual exceeds
   `max_residual_codes`. The prediction must then land inside the support's own
   code range widened by `max_overshoot_codes`, and inside `[min_code,
   max_code]`.
6. **Confidence is applied by the module itself**, so a caller that ignores
   confidence still gets the conservative answer.

Every hole that is not accepted is counted against one of eleven named reasons:
`component_too_large`, `component_too_long`, `insufficient_support`,
`one_sided_support`, `collinear_support`, `incompatible_clusters`,
`support_span_too_wide`, `degenerate_fit`, `residual_too_large`,
`prediction_out_of_range`, `confidence_below_threshold`. The vocabulary is
closed and every reason is always reported, at zero when it never fired, so a
run's conservatism is a stable shape rather than whichever keys were non-empty.

### Every bound is a conservatism knob

Tightening any field of `CompletionPolicy` refuses more and invents less. That
is tested directly, one field at a time: the accepted set can only shrink.

One default is **not** a free choice and is called out in the type's own
docstring. `max_residual_codes` has to come from the sensor's noise floor. The
shipped default of **two raw codes** suits a quantisation-limited frame, where a
plane fit on flat geometry misses its own support by well under one code. A
noisier sensor needs a larger value and pays for it by admitting curvature a
plane cannot represent. It is the one bound a caller must revisit before using
this on real frames.

## Confidence

`CompletedPatch.confidence` is a fraction of 1 derived from named evidence. It
is **not** a camera confidence: it says nothing about the sensor, and no value
of it ever makes a reconstruction a measurement.

It is the geometric mean of five terms, each in `[0, 1]`, so every one of them
can veto and none can be traded away:

| Term | Formula | Direction |
| --- | --- | --- |
| `amount` | `(count - min_support + 1) / (max_support - min_support + 1)` | more supporting samples raises it |
| `spread` | `sqrt(weaker_spread) / radius` | support distributed around the sample raises it |
| `proximity` | `1 - mean_distance / (radius + 1)` | support further away lowers it |
| `fit` | `1 - worst_residual / max_residual_codes` | a worse residual lowers it |
| `agreement` | `1 - code_span / max_support_span_codes` | support disagreeing about the depth lowers it |

The `+ 1` in `amount` is deliberate: the least support the policy accepts is
still support, and scoring it at exactly zero would make the acceptance bound
and the confidence floor the same test twice.

Each term is monotone in the direction its name implies, so the geometric mean
is too. That is tested by varying one term of a fit at a time — varying a real
neighbourhood instead moves several at once, and a test that cannot say which
term moved is not a test of a formula. Thresholded rejection is tested
separately: raising `min_confidence` only ever shrinks the accepted set.

## Work is bounded, and reported

One pass, no iteration to convergence, no queue that can grow. Every hole is
examined at most once and costs at most `(2 * radius + 1) ** 2` neighbourhood
visits, so `CompletionWork.work_budget` is `holes * (2 * radius + 1) ** 2` — a
number derivable from the input **before** the run — and `work_units` is what
was actually spent. Neither is a wall-clock assertion, which would measure the
machine rather than the method.

On the benchmark corpus the spend is well inside the bound: 10,287 of 18,630
units on `plane-fronto`, 81 of 38,313 on `multiview-object`, where nearly every
hole is refused at the component stage before a neighbourhood is scanned at all.

### The ceilings, because a finite integer is not a cap

A per-run budget derived from the policy is only a bound if the policy itself is
bounded, and it is not: `CompletionPolicy(radius=10**9, min_support=8,
max_support=16)` is a perfectly finite, perfectly well-typed policy that
publishes a neighbourhood of 4,000,000,004,000,000,001 samples. Calling it on an
eligible hole would not be a bounded product operation.

So every bound has an absolute product ceiling, refused on construction:

| Ceiling | Value | Why that value |
| --- | --- | --- |
| `MAX_RADIUS` | 32 | four times the widest default disc; a 65×65 neighbourhood is not local evidence about one sample on any sensor this pipeline targets |
| `MAX_SUPPORT` | 256 | the most a least-squares plane in three unknowns is *fitted* to rather than averaged over |
| `MAX_COMPONENT_SAMPLES` / `MAX_COMPONENT_SPAN` | 4096 / 512 | a policy cannot declare the whole frame one recoverable component |
| `MAX_SPREAD` | 1024 | the most any support inside `MAX_RADIUS` can have |
| `MAX_PATCH_DIMENSION` / `MAX_PATCH_SAMPLES` | 4096 / 1,048,576 | four times the 640×400 depth frame the pipeline reads; a dimension larger than this is a corrupted header, and it is refused before the bytes behind it are indexed |
| `MAX_WORK_UNITS` | 25,000,000 | just above the real worst case: a whole 640×400 frame in which **every** sample is a hole costs 256,000 × 81 = 20,736,000 neighbourhood visits at the default radius |

`complete_patch` derives the call's whole worst case — `policy.worst_case_work(holes)`
— **before the first neighbourhood is scanned**, and raises `WorkTooLarge` if it
exceeds `MAX_WORK_UNITS`. That refusal is integer arithmetic on the input, not a
timing observation: it is identical on every machine, and it does not depend on
how long anything took. Each ceiling is tested at its maximum, one past it, at a
huge integer, and with a boolean; the 640×400 default-policy boundary is tested
as the case that must remain admitted.

## Backends

`"python"` is the only implementation and it is the authority; `"auto"` selects
it. Naming any other backend is an error rather than a silent downgrade, on
`scanner3d.compute`'s rule: a caller who selected a backend and got a different
one cannot interpret their own result. `"numpy"`, `"scipy"` and `"gpu"` are
recognised names that report unavailable with a reason. No vectorised backend is
written here — one would need a parity test against this reference first, and
the reference is what the benchmark scores. No dependency was added.

## What the benchmark says, exactly

Run `python -m tests.depth_completeness_harness`. The candidate is
`spatial_completion`, a thin test-side adapter that hands each frame's window to
`complete_patch` and restates the result in the benchmark's provenance
vocabulary. Every number below is from the committed corpus at exit 0.

| Case | Holes | Recovered | Supported (baseline → completion) | False fill | Worst error |
| --- | --- | --- | --- | --- | --- |
| `plane-fronto-complete` | 0 | 0 | 960 → 960 | 0 | — |
| `plane-fronto` | 230 | 107 | 730 → 837 | 0 | 0 codes |
| `plane-oblique` | 225 | 105 | 735 → 840 | 0 | 1 code |
| `curved-surface` | 782 | 0 | 658 → 658 | 0 | — |
| `depth-step` | 195 | 64 | 765 → 829 | 0 | 0 codes |
| `narrow-slot` | 125 | 42 | 835 → 877 | 0 | 0 codes |
| `aperture-ambiguity` | 2 | 2 | 958 → 959 | **1** | 0 codes |
| `multiview-object` | 473 | 0 | 487 → 487 | 0 | — |

- **Complete input is an exact identity.** On `plane-fronto-complete` there is
  nothing to reconstruct and every downstream measurement is the unchanged
  baseline's — not close to it, equal to it.
- **The two planes gain strictly**, and the gain is where a scene-derived bar
  says one is owed. `coverage.local_recovery` requires that every dropped sample
  with a returned measurement on all four sides of it comes back onto real
  surface — 48 samples on `plane-fronto`, 50 on `plane-oblique`, 54 on
  `depth-step`, 34 on `narrow-slot`. The completion clears all four; the
  baseline recovers none of them. Beside `coverage.measured_fraction`, which
  pins the measured population, that is a strict supported-coverage gain.
- **The oblique plane is interpolated, not copied.** Worst error one raw code —
  the truth's own quantisation — against `dense_nearest`'s staircase, on
  identical input. Copying the nearest code cannot produce this.
- **Nothing is invented in empty space on any geometry one frame can settle.**
  False fill is exactly zero in seven of the eight cases. On `curved-surface`
  the background is one 655-sample component and is refused whole. The eighth,
  `aperture-ambiguity`, is where this method loses and is described in full
  below; its one false-filled sample is declared in the fixture rather than
  tuned away.
- **The depth step is neither bridged nor crossed.** `edge.bridging` and
  `edge.misplaced` are both zero while 64 samples are still recovered away from
  the boundary.
- **The slot stays open.** Zero samples reconstructed inside the three protected
  columns, and 42 recovered on the boxes either side of it.
- **Running it per frame costs the multi-view pipeline nothing.** On
  `multiview-object` the fused surface, both tracking records, the poses and the
  dimensions are the baseline's *exactly*, because the completion returns each
  frame unchanged.
- **Confidence is calibrated.** Every reconstruction the completion publishes at
  0.5 or above is right to within one raw code, which is what
  `confidence.high_confidence_worst` gates. The same gate fails
  `dense_nearest`, `broad_interpolation`, `foreground_bias` and
  `easy_samples_only` on the geometries where they are confidently wrong — it is
  a bar with teeth, not a decoration.

### The holes that remain, named

`spatial_completion` **fails `coverage.supported_fraction` in every masked
case**, and additionally `edge_step_column.worst` on `depth-step`. Those
failures are declared in the fixtures rather than explained away, because
conservative refusal is not a success:

- The **four-column strip collapse** on both planes (103 samples), the **edge
  band** at the depth step (125), the **slot** (78), the **silhouette
  background** on the sphere (655) and **every complementary column** in the
  multi-view case (472) are refused at the component stage. In each of them the
  span is further than any support inside one frame reaches.
- On `curved-surface` the sphere's own holes are then refused by the residual
  test: a plane fitted to a curved neighbourhood misses its own support by more
  than two codes, so the case returns the baseline. Recovering it needs a
  surface model this task does not have.
- `depth-step`'s apparent boundary stays displaced by the missing samples. The
  completion does not repair it, because nothing in that one frame says where
  the edge is.

### The ambiguity one frame cannot resolve, demonstrated rather than described

In `narrow-slot` the two box faces either side of the gap are **at the same
depth**, and the truth behind the gap is a plane 44 units further back. Inside
one frame there is no local evidence that distinguishes that gap from ordinary
dropout on a single surface: a fit through the support either side is
well-conditioned, low-residual and completely wrong. That slot is refused only
because it spans the whole frame, so `max_component_span` catches it — which
means it does not actually test the ambiguity at all. It tests a size.

`aperture-ambiguity` tests the ambiguity. Four coplanar slabs at a constant
depth of 100 units tile a 40×24 window except for **one** sample, where there is
genuinely nothing behind the surface: its truth code is zero, and that sample is
real empty space. A `blob` mask of radius 0.5 then removes exactly **one** other
sample, twenty columns away, whose truth is the surface itself. The case carries
no speckle, deliberately, so that the two 9×9 scored neighbourhoods — the whole
of what the default radius-4 disc can see — are **byte-identical**: a single
zero at the centre, code 12800 everywhere else.

One is held-out surface. The other is a hole in the object. A deterministic
single-frame method cannot answer both, and this one does not: it fills both,
because the evidence is the same. So it

- **recovers the dropout correctly** and clears `coverage.local_recovery`, and
- **invents a surface across the aperture**, failing `coverage.false_fill`
  (1 sample against a limit of 0) **at high confidence**, failing the case's
  `confidence.high_confidence_false_fill` gate (1 against 0).

Both failures are written into the fixture as the expected outcome. The
confidence failure is the sharper of the two and the reason the gate exists
here: confidence is derived from the evidence, the evidence is byte-identical,
so the confidence is identical too. **No threshold on this number can separate
the two cases**, because there is nothing in the frame to separate them with.
`dense_nearest`, `broad_interpolation`, `foreground_bias` and `easy_samples_only`
fail identically, which is the point — this is not a property of the algorithm.

#### The epistemic boundary, and what follows from it

This is not a bound that wants tuning. Tightening `max_component_span` until the
aperture is refused refuses the paired dropout with it and gives up the planar
gain, because the two are indistinguishable; that is the trade the reviewer
named, and this corpus now makes whichever side is lost visible instead of
letting a size cap hide the question.

What follows for the rest of the pipeline:

- A `spatially_reconstructed` sample from this ambiguous class — a small,
  locally planar, fully surrounded hole — **must not become tracking input or
  fused TSDF geometry** on the strength of one frame, at any confidence. Its
  provenance and its support are preserved in the result precisely so a later
  stage can tell it apart from a measurement and decline it.
- The evidence that resolves it is independent: another view, a physical
  constraint, or temporal agreement across frames. That is the temporal task's
  job, not this one's.
- Until such a stage exists, this module's output is derived evidence beside the
  raw frame and nothing consumes it — see the section below.

## Adversarial controls

Beside the benchmark's existing nine corruption controls, two new ones corrupt a
**real** completion result and must earn an exact refusal:

| Control | Corrupts | Must earn |
| --- | --- | --- |
| `spatial_completion_forged_work` | reports one refusal as an acceptance, changing nothing else | `work_counts_inconsistent` |
| `spatial_completion_recursive_support` | cites one of its own reconstructions as the evidence for another | `support_not_measured` |

`work_counts_inconsistent` is a new refusal in the contract. A work record is
the only part of a candidate's account that is about the **run** rather than
about the output, so every field of it is bound to something the oracle counts
for itself: the holes are the input's own no-return samples, the accepted count
is the reconstructions the result carries, the support total is the support it
named, and the budget is the policy's own bound for a run of that size.

At the module level, `PatchUnusable`, `PolicyUnusable` and `ResultUnusable`
refuse a patch that is not its own rectangle, a code that is not a uint16, an
unbounded or non-finite or boolean policy value, a support requirement its own
radius cannot hold, a second pass, a truncated result, a non-finite confidence,
a reconstruction with no named support, a reconstruction cited as support, and a
work record that does not add up.

Beside those, the direct-forgery controls in `tests/test_depth_completion.py`
take a **real** result from `complete_patch` and change exactly one published
fact with `dataclasses.replace`, which is the form the review used: a
reconstructed code, a confidence, the published policy, one support sample
added, omitted or exchanged, `work_units`, the rejection taxonomy, a refused
hole shown as recovered, a recovered hole hidden as refused, and a boolean or
float `width`, `height` or `schema_version`. Each one names the exact refusal
it must earn.

## What this does not change, and why

- **No capture, GUI, preview, tracker, TSDF, mesh or export path calls this
  module.** A reconstruction is evidence of a different kind from a measurement,
  and averaging the two without saying so is how a scan quietly stops being one.
  Each of those stages has to decide what a reconstructed sample means to it —
  weight in fusion, eligibility as an ICP correspondence, how the preview draws
  it — and that is a per-stage decision, not one this task can make for them.
- **No temporal completion.** Every reconstruction here names support inside its
  own frame. What another view saw is a different problem with a different
  failure mode, and it is a separate module —
  [docs/depth-temporal.md](depth-temporal.md). That stage may confirm an
  estimate made here, contradict it and publish its own measurement instead, or
  never reach it at all; an estimate it never reached keeps the spatial
  provenance this module gave it, unchanged.
- **No scanner setting, hardware access, dependency, CI job or network action.**
  Nothing in this task read a capture or touched a device.
- **No physical claim.** Every number above is in synthetic units of raw code
  times depth scale, over scenes rendered by `scanner3d.simulate`, which is not
  a sensor model — no speckle physics, multipath, subsurface scattering or
  baseline shadowing. Passing this benchmark is necessary and not sufficient.
  Physical evidence is a separate, later gate and this task produced none.

## Running it

```bash
.venv/bin/python -m pytest tests/test_depth_completion.py
.venv/bin/python -m pytest tests/test_depth_completeness.py tests/test_depth_completeness_harness.py
.venv/bin/python -m tests.depth_completeness_harness --out /tmp/depth-completeness.json
```

The harness is the authority. Two runs of it are byte-identical, and a report it
publishes is read back through its own strict reader, which now also holds the
confidence distribution and the work record to the coverage record beside them.
