# Tracking: how a frame is placed, and when it refuses to be

Status: implemented and shared by the live capture and the offline
reconstruction. Every ordinary frame is placed by agreement between independent
references rather than by one alignment against its neighbour, every placement
carries the evidence it was judged on, and a frame no reference can place is
refused rather than given the last pose. Nothing here is a promise that a pose
is *correct* — the last section says what is still not measured.

## The capture this exists because of

`/home/ben/scans/scan/20260828-105933`, on 2026-08-28, is the same capture
[docs/tracking-loss.md](tracking-loss.md) is written around, and it is the
reason for this layer too. Tracking-loss detection made that failure *visible*.
Support-plane suppression ([docs/support-plane.md](support-plane.md)) removed
one plausible cause and, being conservative, applied to none of its 176 frames.
Neither made the scan work, because neither addressed what was actually wrong:

**a pose came from one pair of frames and one pair only.**

Carrying the scanner around the mapper chassis breaks that in two ways at once.
Individual views go weak — pointed at one flat panel, a frame has no geometry
determining rotation about that panel's normal, and point-to-plane alignment
does not report the singularity, it reports a small residual for a transform
that slid. And nothing corrects the chain: a pose is the composition of every
earlier relative fit, so every error in it is permanent and invisible from
inside.

## What replaced it

`src/scanner3d/tracking.py` — one pure, deterministic, bounded module, used by
the window's ordinary capture, by the tentative chain of a relocalization, and
by `reconstruct_sweep`. For each ordinary frame it forms up to three
**independent** candidate placements and then decides between them:

| Candidate | Reference | Why it is there |
| --- | --- | --- |
| `adjacent` | the immediately previous accepted frame | the most direct measurement there is, when it works |
| `keyframe` | one accepted frame chosen for its viewpoint, nearest the predicted pose | a second opinion from somewhere the previous frame is not |
| `model` | a bounded sample of several accepted frames, in accepted-model coordinates | the only reference broader than one view, and therefore the only one that does not accumulate drift |

Every candidate is a **refinement from a stated initial transform**. The
adjacent solve runs first; when it produces a finite transform, the pose it
implies is the starting point for the other two. Nothing here searches for where
a frame might be — that is a relocalization
(`src/scanner3d/relocalize.py`), a different and much more expensive decision,
and this module never performs one.

## Where the starting pose comes from

A refinement is only as good as the place it starts, and this scanner delivers
about 4.5 frames a second (`src/scanner3d/v4l2_stream.py`). Between two frames a
carried scanner has genuinely moved, so the two statements available about where
the next frame is are:

- **the scanner did not move** — the identity, relative to the last accepted
  frame;
- **the scanner kept doing what it was doing** — the step it was last measured
  making.

Only the second is ever true on a moving capture, and until this existed the
tracker used only the first. It seeded the adjacent solve at the identity, and
where that solve had not produced an accepted placement it seeded the keyframe
and model solves from the previous *pose* — which is the same assertion again,
made on exactly the frames that had already failed to be placed once.

The tracker now keeps one **constant-velocity estimate**: the step between the
last two accepted frames, and nothing else. There is no IMU on this unit to
measure motion with — `docs/illumination-actuators.md` records that the IIO
device the firmware expects is simply absent — so this is the only motion
evidence available, and it is spent in two places:

| Where | What changes |
| --- | --- |
| the prediction the keyframe and model solves are seeded from | starts at `previous_pose ∘ velocity` instead of `previous_pose`, for no extra solves at all |
| the adjacent candidate | may be retried **once** from the prediction, and only where the identity-seeded attempt produced no accepted placement |

**The identity attempt stays first, and that is a constraint rather than a
preference.** `reconstruct_sweep` pre-fits every adjacent pair up front and
across worker processes, before any pose chain exists to extrapolate a velocity
from. A tracker that seeded its *first* adjacent solve from a prediction could
no longer accept those pre-fits, and the offline path would have to become
sequential to keep agreeing with the live one. Keeping the identity attempt
first is what keeps a run with workers, a run without, and the capture that
produced the frames on the identical chain.

**The prediction changes no gate.** It decides which local minimum a refinement
can reach and nothing else. A frame it seeds badly is refused by the same
residual, overlap, travel and agreement evidence as any other, a frame with no
usable prediction is placed exactly as it was before this existed, and the retry
can only turn a refusal into a placement — where the first attempt was accepted,
the retry never runs and no solve is spent. Which seed reached a candidate is
recorded on it, because "the previous frame placed this" and "the previous frame
placed this once it was told roughly where to look" are different statements
about how a capture is going.

### When there is no prediction

Deliberately absent, rather than approximated, wherever it would be a guess:

- **no continuous chain.** A velocity is recorded only for an ordinary frame
  that arrived with nothing refused behind it. An anchor has no predecessor, a
  seeded relocalization bridge crossed a seam that is not motion at all, and an
  acceptance with refusals behind it spans a gap whose size the next frame
  cannot divide back out.
- **more than three delivery steps since the last acceptance.** The tracker
  numbers frames by *acceptance*, so a refused frame leaves the next one two
  steps from its reference, the one after three. The prediction is composed once
  per step for exactly that reason — and past three steps the scanner has been
  unplaceable for the better part of a second, which is the tracking-loss
  streak's own territory, and every further step multiplies whatever the last
  one got wrong.
- **already past the travel gate.** A fit that stayed near such a seed implies
  the surface jumped, which the travel gate refuses whatever produced it. So the
  seed is withheld rather than made and then refused.
- **inside the stationary band.** Then it is not a different starting point from
  the identity the first attempt already used, so a retry would re-run that solve
  and reach that answer again, and the frame is decided on one solve.

  Note what this band is *not* about. It measures the step between two frames,
  not whether the scanner is bolted down — so a turntable does **not** land in
  it. There the scanner is fixed and the object turns, which means consecutive
  frames differ by the whole rotation step and a prediction is formed exactly as
  it is for a carried scanner. Measured on `20260901-145316`, a turntable
  capture: travel of 1.77 to 10.60, median 3.38, which is the middle of the
  guidance band. Only a scanner pointed at something genuinely not moving is
  inside this band.

Both of the last two bounds are the module's existing numbers — the travel gate
and the keyframe stationary band — rather than new ones invented here.

### What it is measured to do

On this repository's synthetic scene, 16 frames at each step, with the
*identical* clouds given to both policies. The only difference is whether a
refused adjacent solve may be tried again from the prediction. "Surface travel"
is the median distance the frame's own points move between consecutive frames,
which is the quantity `travel_verdict` judges a capture's speed by; its "too
fast" ceiling is 5.85.

| Step per frame | Surface travel | With the prediction | Identity seed alone |
| --- | --- | --- | --- |
| 7°–18° | 2.3–5.8 | all placed | all placed, identically |
| 30° | 9.6 | 16/16, worst error 2.32° | refuses at frame 7 |
| 33° | 10.5 | 16/16, worst error 6.87° | 16/16, **41° wrong** from frame 7 |
| 36° | 11.4 | 16/16, **23° wrong** from frame 10 | refuses at frame 6 |
| 39° | 12.4 | 16/16, worst error 2.13° | refuses at frame 6 |

A caution on reading that "surface travel" column, measured afterwards on a real
turntable sweep (`20260901-191525`, constant motor rate, 60 frames). Travel is
the product of three things — the angular rate, how far the visible surface sits
from the axis, and the gap between kept frames — and only the first is the
motion. Across that sweep travel varied **6.0×** (CV 37%) while the recovered
rotation angle varied **3.0×** (CV 18%), and dividing out the camera's own
alternating 69/373 ms delivery left **13.9%**. So travel is the right quantity
for "how far did the surface move relative to the noise", which is what
alignment cares about, and the wrong one for "how fast is this turning".

Read those rows carefully, because they do not all say the same thing.

Inside and somewhat beyond the guidance band the prediction is free and
invisible: at the ordinary step the identity seed is already close enough that
the two policies place every frame identically. Past it — the 30° row, at
roughly 1.6× the surface travel `travel_verdict` already calls too fast — it
converts a sweep that gives up a third of the way through into a complete one,
at the same accuracy, and the 33° row is the same mechanism catching a chain
that the identity seed alone drove 41° wrong while reporting success.

**And the 36° row is the honest cost.** There the prediction carried a capture
that would otherwise have refused, to 23° of error. It is not a trend — 39°
returns to 2.13° across all 16 — because what changes between those rows is
which local minimum one solve falls into, and a seed cannot make a refusal
safer. It can only decide where a solve starts. Past the guidance band, a
placement this module accepts is worth no more than the geometry it was measured
on, and both policies produce confidently wrong chains there.

The operating conclusion is unchanged by any of this: **slow down**. The
prediction widens the speed at which a capture survives; it does not move the
speed at which a capture is good.

[docs/frame-spacing.md](frame-spacing.md) is the other answer to the same
complaint, and it is a different mechanism entirely: it changes which deliveries
are offered for placement rather than where a placement is started from, and it
holds the measured travel inside the guidance band instead of widening what
survives outside it. Neither touches a tracking gate.

### The travel gate is judged against the prediction (2026-09-04)

The gate that refuses a placement as "a jump to somewhere else on the object"
measured the surface travel since the last ACCEPTED frame, and that made it the
speed limit for a hand-carried scanner. Replaying the turntable capture
`20260904-020845` at larger frame strides:

| step per frame | median travel per accepted frame | outcome |
| --- | --- | --- |
| 2.8-11 deg (strides 1-4) | 1.3-5.2 | every frame placed |
| 17 deg (stride 6) | 7.4 | 22 of 26 placed; three disagreements |
| 23 deg (stride 8) | - | ICP placed the first step at 15.706 against the 15.600 gate; refused, and the next frame was two steps away; lost |
| 34 deg (stride 12) | - | placed at 24.3 and refused the same way |

A speed-ramp built from the same frames placed 30 frames through 20 degrees
per frame and lost the capture at the first 23-degree step: one travel refusal
at 16.606, then two residual refusals at the doubled gap. The hand-carried
capture `20260828-105933` ended the same way at `frame_0070`, 20.662 against
15.600. In every case the solver had found the frame; the gate declined it.

So the gate now has a second reading. Where a motion prediction exists (the
last accepted step carried forward, see above), a placement is refused as a
jump only when it is far from the last accepted frame AND far from where the
prediction put it. A scanner carried steadily moves the surface a long way
between frames and almost none from its own prediction; a solve that locked
onto somewhere else is far from both. The record carries the second number as
`travel_from_prediction` beside `travel`, and the schema version moved to 6.
The prediction itself may now exceed one gate, and is discarded past three
(`MAX_PRIOR_TRAVEL_MULTIPLE`): 46.8 units in one delivery step is about 21 cm/s
at the measured 4.5 fps, where a fringe-projection frame is motion-corrupted
before any tracker sees it. A first fast step with no measured velocity is
still a jump, so a capture has to start moving before it moves fast - which is
what a hand does.

Replayed on the new gate: the speed ramp runs to its end, 36 of 38 frames
placed through 23 and 28 degrees per frame (13.2 units of travel on the longest
accepted step, where the old gate stopped at 15.6 and the ramp died at 16.6),
and the last two frames at 34 degrees are refused by reference DISAGREEMENT,
12-13 units apart, not by travel. The constant-stride replays at 23 and 34
degrees are unchanged, because their first step has no velocity behind it.
The hand-carried capture places 67 frames where it placed 65 - `frame_0056`
and `frame_0064` are now steps rather than jumps - and still ends at
`frame_0070`, whose placement sits 20.662 from the last accepted frame and
31.616 from the prediction: a jump by both readings, preceded by two residual
refusals at 2.73 and 2.63 against 2.60. Past the travel gate, the residual gate
and reference agreement are the limits, and both measure the frames
themselves.

What this does not change: the residual, overlap and agreement gates, the
delivery rate the hardware sets (4.5 depth frames a second), and the window's
default of keeping one frame in three, which triples the motion per kept frame
and is the wrong setting for a hand-carried sweep.

## What "accepted" is allowed to mean

Not "ICP returned". Each candidate carries the four measurements this repository
already judges an alignment by — finiteness, convergence, residual against the
measured noise floor, matched and used fractions — judged by the *same function*
the tentative local chain uses, `relocalize.judge_local_edge`. Two more are
added because this decision needs them:

- **overlap** with the reference the residual came from, because a small
  residual over a corner of a frame is not evidence about the frame;
- **travel**, the distance the surface moved since the last accepted pose,
  because a placement can be finite, converged and well matched and still
  describe the scanner arriving somewhere else entirely.

Then the surviving candidates are compared, and this is the part that matters:

- **Agreement is the evidence, and it has to be mutual.** The surviving
  candidates are grouped into the largest sets in which *every pair* is within
  the gate of *every other pair*. Comparing each survivor against a leader is
  not enough: two placements nearly twice the gate apart, each just inside the
  gate of a third, are not an agreement.
- **The agreement has to be unambiguous.** Usually that means exactly one
  largest mutual-agreement group. If largest groups tie, one may win only when
  its widest internal pose disagreement is at least `1.15x` tighter than the
  runner-up's — the same ambiguity margin the repository uses elsewhere.
  Tightness is measured between the candidate poses on the frame's own surface,
  not from either ICP residual. Comparable groups remain two answers and are
  refused.
- **A majority is required; a lone dissenter is outvoted, not obeyed.** With the
  shared policy's three candidates: two agreeing and one dissenting places the
  frame, one agreeing and one dissenting does not. The dissenter is named in the
  record (`dissenting`) rather than hidden. This is the rule that keeps a
  capture alive through one badly seeded solve, and the reason it exists is
  mechanical: the adjacent candidate is started from the identity relative to
  the last **accepted** frame, so one refused frame leaves it a whole skipped
  step away from the answer — the condition `icp_plane`'s own linearisation says
  it does not survive. Measured on the owner's 2026-08-28 capture, frame 10: the
  adjacent candidate cleared every gate and sat 5.59 and 5.54 from the keyframe
  and model candidates, which sat 1.75 from each other. Letting it veto them
  ended the capture there, and ending it there cascaded — the next frame's
  adjacent reference was two skipped steps stale, then three, and the matched
  fraction never recovered.
- **The pose comes from the broadest reference in the agreeing group** — the
  model first, then the adjacent pair, then a keyframe. The order is measured,
  not assumed: preferring the model holds the error flat over a sweep where every
  other choice lets it climb like a chain's, and preferring the adjacent pair
  over a keyframe places an early frame 0.2 degrees out where the keyframe places
  it 1.2 degrees out.
- **A `model`-only survivor is always a rescue.** Whenever the model candidate
  is the only one that survived — whether the others were refused, could not be
  formed because ICP raised on a singular panel, or did not exist yet — the
  reference must span several accepted viewpoints and the frame must overlap it
  well, because a reference assembled from one or two nearby viewpoints is blind
  in the same direction a weak frame is. There is no start-of-capture exemption:
  a capture begins with an anchor, which is placed without being fitted at all.

### What the record calls support

Three separate fields, because they are three separate statements and round 1
published only one of them under the strongest name.

- `agreeing` — the **selected consensus cluster**: the candidates that agreed
  with each other and placed the frame. `origin` is always one of them. An
  anchor reports `["anchor"]` and a seeded bridge frame reports `["bridge"]`,
  so every accepted decision names its support the same way.
- `passing` — every candidate that cleared its **own** gates, agreement aside.
  A superset of `agreeing`: a reference can be finite, converged, inside the
  residual and well overlapped and still put the frame somewhere the others do
  not.
- `dissenting` — the reference that cleared its gates and then contradicted the
  cluster. Never more than one; two would be a refusal.

A refused frame has an empty `agreeing` and may still have a populated
`passing`, which is exactly what a disagreement looks like.

The tie margin closes one measured false refusal without weakening any
candidate gate. On physical capture `20260831-175352`, frame 88 had three
individually acceptable placements. Adjacent/keyframe were `1.620` apart and
adjacent/model were `4.760` apart, while keyframe/model were outside the
unchanged `5.200` agreement gate. Treating those two size-two groups as equally
strong refused frame 88 and made frames 89 and 90 progressively harder to seed.
The pair diameters are `2.94x` apart, so adjacent/keyframe is a bounded fallback
and model is its dissenter. A symmetric pair or a near tie still refuses;
neither residual is used to select one.

That fallback does not outrank stronger evidence. The tracker first tries its
bounded recovery against two disjoint multi-frame references. On frame 88 the
coarse ranking itself was clear (`1.683x` shared score margin and `2.154x`
whole-frame coverage margin), but the second reference overlapped the proposed
placement by `0.54`, just under the unchanged `0.55` confirmation gate. Recovery
therefore refused and the adjacent/keyframe fallback placed the frame, carrying
that failed recovery evidence on its record.

A read-only replay of all 91 frames with the appliance's default `auto` compute
backend accepted 91/91: frame 88 used that fallback after one failed recovery
attempt, and all three references agreed again on frames 89 and 90. No tracking
loss was reached. This establishes that the terminal loss was created by the
old tie decision; it does not independently establish that any accepted pose is
geometrically correct.

### The convergence reading, stated plainly

Convergence means the solver's residual stopped changing between iterations by
more than a threshold, and the whole question is what that threshold should be.
`icp_point_to_plane` expresses it as a fraction of the **target's extent**, and
its 1e-6 default is about 6e-5 on a 66-unit object — a thousandth of a
thousandth of the sensor's own noise. Measured on this repository's own 24-frame
sweep at that default: residuals of 0.06 to 0.18 with matched and used fractions
at their gates, and frames still reporting `converged=False` at fifty
iterations, because the trimmed correspondence set reshuffles slightly on every
pass. Refusing all of those would end a healthy capture in three frames.

The fix is to state the threshold in the units the decision is made in. A
residual is a distance; the smallest distance that means anything on this sensor
is the noise floor; so the tracker **runs** the solver at a threshold of a
hundredth of the noise floor — 0.0065 depth units, the same absolute threshold
whatever the object's extent, recorded in the policy as
`convergence_threshold`. On the same sweep every frame then converges in five to
nine iterations at residuals matching the 1e-6 run to three decimals: the same
fits, honestly reported as settled.

Convergence is therefore an ordinary hard gate here, exactly as it is for the
tentative chain. A candidate that never settled is refused, is not counted among
the references that agreed, and cannot place a frame — including as a rescue.

## A refusal changes nothing

A refused frame appends no pose, does not touch the keyframe set, does not slide
the retained clouds and does not move the model reference. That is load-bearing
rather than tidy: the live capture builds its failure streak from these
refusals, and a tracker that had half-mutated itself would make the next frame's
evidence describe a model no decision ever accepted.

Live, a refusal enters the existing streak with the tracker's own account of the
frame — which references were tried and what each said — and three in a row
reach the existing durable tracking loss, unchanged. Offline,
`reconstruct_sweep` **refuses the whole sweep**, naming the frame index and
carrying the complete placement evidence, rather than skipping the frame (which
silently renumbers everything after it) or manufacturing a pose for it (which
produces a chain that looks complete and is wrong exactly where the evidence ran
out).

## Keyframes: motion and novelty, never frame number

A scanner left pointing at one spot delivers frames at exactly the rate a moving
one does, so a keyframe set filled by counting frames holds twelve copies of one
view precisely when the tracker most needs twelve different ones. A frame
becomes a keyframe when its pose sits further from *every* existing keyframe
than the motion gate, or when it overlapped the model reference poorly while
still having moved out of the stationary band — rounding a corner moves the
scanner barely at all and changes what it can see completely.

The set is bounded. The anchor is never evicted, because it defines the model
frame; the newest two are never evicted, because they are where the scanner is
now; among the rest the one closest to another keyframe goes, because it carries
the least coverage the set does not already have. Ties go to the lowest index,
so the answer does not depend on iteration order. Every insert, retain and evict
decision is recorded with the measurement behind it: "the set holds 0, 4 and 9"
does not say why 6 is absent, and the difference between "it was never novel"
and "it was evicted as redundant" is the difference between a capture that
sampled its viewpoints badly and one whose bound is too small.

## Bounds

Everything is bounded, and the bounds are a schema-versioned policy
(`TrackingPolicy`) that reaches the durable record, so a reader comparing a
capture's decisions with a reconstruction's can *see* the two ran the same one
rather than assume it.

| Bound | Value | What it stops |
| --- | --- | --- |
| ICP solves per offered frame | 3, or 4 where the adjacent solve is retried from the motion prediction | per-frame cost growing with capture length |
| keyframes held | 12 | the reference set growing with capture length |
| accepted frames per model reference | 8 | the reference build growing with capture length |
| points in a model reference | 1800 | the model candidate's solve growing with the model |
| points any candidate is fitted FROM | 900 | a dense offline frame costing more than a live one |
| points kept per retained cloud | 900 | a reference costing whatever the caller happened to offer |
| accepted clouds retained | 32 | memory growing with capture length |
| coarse searches per offered frame | 1, and 0 unless the frame was refused or a tied consensus needed stronger arbitration | recovery cost growing or leaking into an unambiguous ordinary placement |
| initial poses one coarse ranking holds | 26 — 25 generated orientations and 1 prior | a search growing with the number of things worth trying |
| ICP solves per recovery attempt | 9 | the recovery's cost growing with the model |
| accepted frames one recovery inspects | 8 | the recovery's references growing with capture length |


Every one of those is a bound on the work the tracker actually does, which it
was not in round 1: the adjacent candidate was exempted from the point budget
"because it is the solve the capture has always done", so a physical replay
reached about 18000 points against a nominal 900 and took 738.971 seconds for
176 frames. The same replay now runs at about 1.04 seconds a frame. The offline
path samples the pairs it pre-fits exactly the same way (`bounded_source`,
`retained_cloud`), so the live and offline solves remain the same problem.

`TrackingPolicy` also refuses combinations whose bounds cannot all hold at once
— a keyframe bound the anchor and the protected recent tail already fill, a
retention budget too small to hold every keyframe and the previous frame, a
reference point budget smaller than the number of frames it would name. A
tracker that can be constructed is one whose hard bounds are reachable.

Poses are kept for every accepted frame — they are the answer, and they are
small. Clouds are kept only while they can still be a reference.

## When ordinary references do not produce one unambiguous answer

Three of the refusals above — nothing survived, the survivors disagreed, an
unsupported model-only rescue — are offered, once, to a bounded coarse-to-fine
fallback before they become the frame's answer. One accepted case is offered
too: equally large consensus groups whose distinctly tighter member is a
bounded ordinary fallback. The disjoint multi-frame recovery is stronger
evidence, so it gets the first chance; if it cannot confirm a replacement, the
ordinary fallback remains and carries the failed recovery evidence. Nothing
else reaches this path: an empty or malformed frame is refused before a
candidate exists, an unambiguous ordinary placement never reaches it, and a
frame every reference that aligned it at all called a *jump* is declined without
a search, because a better starting pose cannot move the last accepted pose the
travel gate measures against.

The problem it is for is specific. When every reference refines from a stale
prediction into a *different* local minimum, the frame is refused while the
geometry is still there — which is what the physical capture below does at
`frame_0032` onwards. A coarse search re-seeds the refinement from a shape-level
comparison instead. It changes where refinement starts and nothing else.

What it has to establish before it may produce a pose:

- **Two reference sets that share no accepted frame**, each dealt alternately
  across the whole capture so both span the same range of viewpoints, and
  neither containing the previous accepted frame whose prediction just failed.
  One reference cannot confirm itself.
- **Measured viewpoint diversity in each of them.** Distinct frame indices are
  not distinct viewpoints: an operator who stopped moving produces four accepted
  frames that are four readings of one view, and a reference built from them is
  blind in exactly the direction the offered frame is. So the largest keyframe
  pose gap inside each partition is measured — the tracker's own
  `_pose_distance`, degrees over the keyframe motion gate and distance over it,
  whichever is larger — and a partition under one whole gate is refused by name,
  with its measured span on the record either way.
- **Every rescue gate, inherited whole.** An ordinary candidate is not the
  independent confirmation for a recovered placement: this path was reached
  because the ordinary evidence either refused or split between competing
  groups. The recovery therefore clears the same no-second-opinion
  `rescue_min_reference_frames` and `rescue_min_overlap` gates on the confirming
  partition as well as the searching one. A coarse search contributes a starting
  pose; it contributes no evidence about how many viewpoints the reference spans
  or how much of the frame is on it.
- **Three-dimensional coarse evidence.** A frame or reference that does not span
  three dimensions determines no orientation and is refused before ranking.
  Principal extents within `1.25` of one another are different: they mean the
  PCA-derived axes are weak seeds, not that the shape's details or the explicit
  last-pose prior cannot distinguish placements. That warning remains in the
  record, while the shared score margin, whole-frame coverage margin, ordinary
  gates and disjoint-reference confirmation below decide whether a pose exists.
- **The last accepted pose, offered to the ranking as a prior.** A recovery is
  reached for a frame whose references disagreed, not for a frame that is lost,
  and the last accepted pose is the one placement in the capture this module
  already trusts — it is what the travel gate measures the result against. It is
  handed to `coarse_align` as one more initial pose and scored by the same
  decimated pass on the same grid as the 25 generated orientations. It buys no
  confidence: it wins the ranking or it loses it, and it clears every gate the
  others clear. What it buys is a starting pose whose centroid is where the
  frame actually is — measured on the capture below at `frame_0029`, the best
  generated orientation covered the recovery reference 0.10, because a partial
  view's centroid is not a multi-frame reference's centroid, while the last
  accepted pose covered it 0.46.
- **TWO unambiguous verdicts, both required, neither substituted.**
  `coarse_align`'s own verdict must call the ranking unambiguous *and* the
  coverage margin must clear `multiview.AMBIGUITY_MARGIN` against the best
  genuinely different hypothesis. Round 1 recorded the shared verdict and
  accepted while it was false; that is withdrawn. The shared ranking was
  corrected rather than worked around: its runner-up is now the best candidate
  that actually places the source *somewhere else* — further than the scoring
  pass's own two-cell neighbour reach — instead of whatever ranked second, which
  was routinely one placement spelled twice. Both margins, both verdicts and
  `coarse_align`'s limitations are on every recovery record, accepted or not.
  **What requiring it used to cost, and what was corrected.** `coarse_align`
  scored a candidate on the mean nearest-neighbour distance over the points it
  *matched*, so unmatched geometry contributed nothing: a placement that put
  most of the frame off the reference was scored on the fraction of itself it
  managed to place. On the incompatible-minima fixture the right placement and a
  genuinely different one scored within 1.05× of each other while covering the
  reference 0.47 against 0.34, and the recovery refused a frame the geometry
  distinguishes clearly. The correction went into the shared stage, not around
  it: the score is now a **truncated mean over all the sampled points**, a point
  with no partner costing a bounded `SCORE_MISS_CELLS` grid cells rather than
  nothing. Bounded, because a partial view legitimately has points the other
  view never saw and an unbounded penalty would rank on overlap alone; and at
  `2·√3` cells, the furthest apart two points can be and still be found by the
  scoring pass's own 3×3×3 neighbour search, so no matched point can ever cost
  more than a miss and the mean is monotone in coverage. The same fixture now
  reports a shared margin of 2.42. That change moves `relocalize`, `revisit`,
  `support` and `quality` rankings too, which is the point: it is a correction
  to the shared measurement, not a tracker-private substitution for a verdict
  this module did not like.
- **The margin belongs to the hypothesis that was selected.** The coverage
  margin is measured before refinement, from the ranking's best, and *again*
  afterwards from the hypothesis actually admitted, against every genuinely
  different placement still standing beside it — including the higher-ranked
  ones a gate refused, which therefore refuse the recovery rather than lending
  it their confidence. Only a hypothesis whose refinement landed on the selected
  placement is excluded, because two seeds reaching one answer are one answer.
  On the owner's capture, round 1 let a rank 0 that failed travel donate its
  3.317 margin to the rank 2 it then confirmed; that cannot happen now.
- **One placement, agreed.** Several coarse seeds collapsing into one pose is
  refinement working; two survivors in two places are two answers, and the
  smaller residual arbitrates neither. The surviving placement is then refined
  against the independent partition and must agree with it inside the same gate
  the ordinary decision uses.
- **Every ordinary gate.** Convergence, residual, matched, used, overlap and
  travel are the same measurements on the same thresholds. Recovery improves the
  basin; it widens nothing.

Every count is recorded beside the bound it is checked against, and the coarse
and refinement stages are counted separately because they are different work on
different data: the ranking is scored from a bounded sample of the frame, each
refinement is fitted from the larger sample every ordinary candidate uses, and
the search ranks every orientation it generates however short a shortlist it is
asked to score for coverage.

A recovery that refuses leaves the ordinary refusal exactly as it was — same
failure type, same words, same absence of mutation — so the consecutive-refusal
streak, the durable tracking-loss record and the post-loss relocalization state
machine behave as they did before this fallback existed. Live status reports a
coarse recovery as its own thing, not as an ordinary model placement.

## Recorded relocalization boundaries

A boundary decided during a capture is **seeded, never refitted**. The two
frames either side of it are not a small step apart, and handing ICP such a pair
is the one thing the boundary exists to prevent. The offline reconstruction
drops the recorded transform in where the pair would have been fitted and
continues tracking from there against the same accepted model. The boundary
frame itself always becomes a keyframe, because it is where the model gains a
view it has no other way of reaching; the frames published behind it are
ordinary members of the chain, so one published batch cannot fill the whole
keyframe set with one viewpoint.

## Measured, on synthetic geometry

`tests/test_tracking.py` walks a camera round an asymmetric object at 22.5
degrees a step for 24 frames and places it twice from the identical clouds and
the identical ICP settings — once by blind adjacent chaining, once by this
module. Rotation error against the known truth:

| | frame 1 | frame 8 | frame 16 | frame 24 |
| --- | --- | --- | --- | --- |
| blind chaining | 0.42 | 0.56 | 1.11 | **1.60** |
| tracked | 0.42 | 0.53 | 0.78 | **0.84** |

The endpoint ratio is 1.90x and it grows with the length of the sweep, because
one of the two numbers grows and the other does not — which is the mechanism,
not a tuning. The same tests require the blind chain's second half to be worse
than its first and the tracked chain's not to be, so a scene that was simply too
easy would fail rather than pass quietly.

The tracked figures are wider than the ones this document carried in round 1
(0.62 worst, 2.75x), and the reason is a correction rather than a regression.
Round 1 exempted the adjacent candidate from the point budget and built the
model reference out of the keyframes the keyframe candidate was already using.
Making both true — every candidate fitted from 900 points, and the model
reference built only from accepted frames no other candidate is using — costs
precision on this fixture, and it was measured which of the two costs it: with
the same disjoint references and a 4000-point budget the sweep returns to 0.89
worst and 0.66 final. 900 points is what the live capture actually has, so this
is the honest number for the appliance and the narrower one was not.

## Reading a real capture

`tests/tracking_replay.py` replays a physical capture's raw frames through the
tracker and prints, per frame, which candidates were formed, which agreed, which
reference placed it, and the exact refusal when none did. It is strictly
read-only — `glob`, `is_file`, `stat`, `read_bytes` and nothing else — and it is
not a pytest module, because a residual measured against one physical capture is
not a bound that means anything against another.

```bash
.venv/bin/python -m tests.tracking_replay --capture /home/ben/scans/scan/20260828-105933
.venv/bin/python -m tests.tracking_replay --frames 40 --json
```

It replays the **exact live preparation** — `preparation.prepare_frame` at
`gui.PREVIEW_POINTS`, so the same deprojection, clip, thinning, speckle
rejection and support-plane suppression the appliance runs — and it stops where
the appliance stops: once three consecutive refusals reach
`capture_sequence.TRACKING_LOSS_STREAK`, that loss is terminal, no later frame
is offered to the tracker, and nothing after it can be reported as a
reacquisition.

### Fixed scanner, spinning target: `20260831-202514`

This 38-frame physical capture lost tracking while the scanner remained fixed
and the printed target was spun. The depth feed did not disappear: raw return
yield was `24.9%`, `26.5%` and `34.6%` on frames 35–37, and the target remains a
large structured surface in each depth image. Support-plane suppression applied
to none of the 38 frames, so the registration clouds conservatively retained
whatever support and background lay inside the depth clip.

The ordinary references became unstable before the terminal loss. Frames
31–34 alternated between adjacent/keyframe and keyframe/model consensus. At
frame 35 all three candidates still cleared their own residual, correspondence,
overlap and travel gates, but put the surface `7.183`, `10.114` and `17.095`
apart pairwise against the unchanged `5.200` agreement gate. Refusing that
ordinary decision was correct; no pair agreed.

Two policy defects turned that instability into a false loss. First, a
distinctly tighter member of tied equal-size consensus groups was accepted
immediately even though disjoint multi-frame confirmation is stronger evidence.
Second, when recovery was eventually reached at frame 35, one reference's two
principal covariance extents were within `1.25`; the old code treated weak PCA
axes as proof that every ranked placement was ambiguous and refused before one
refinement. Those are different facts.

Recovery now arbitrates the first bounded tie, at frame 31, and a near-equal
principal extent remains a warning rather than a veto. Every measured gate is
unchanged:

| frame 31 recovery evidence | measured | gate |
| --- | ---: | ---: |
| shared coarse score margin | `2.108x` | `>= 1.15x` |
| whole-frame coverage margin | `4.092x` | `>= 1.15x` |
| selected placement's own coverage margin | `4.222x` | `>= 1.15x` |
| disjoint-reference pose disagreement | `0.169` | `<= 5.200` |
| ICP solves | `9` | `<= 9` |

The searching reference is marked principal-frame-degenerate, but its ranked
geometry is not ambiguous and the second disjoint reference independently
refines to the same placement. The result clears the same convergence,
residual, matched, used, overlap and travel gates as an ordinary candidate. A
read-only replay with the appliance's `auto` backend accepts 38/38: one bounded
recovery at frame 31; unanimous ordinary agreement at 32 and 33;
keyframe/model agreement at 34; adjacent/keyframe agreement at 35; and adjacent
placements at 36 and 37. A genuinely symmetric cube remains refused before
refinement because its actual shared score margin is `1.00x`; removing the
blanket warning-as-veto does not remove the measured ambiguity gate it was
duplicating.

As with every physical replay here, this establishes why the software declared
a loss and that its own independent evidence supports a continuation. It does
not supply surveyed pose truth for the accepted continuation.

### Hand-carried scanner: `20260828-105933`

**The expected honest result on that capture is not "every frame is
recovered".** It is the tracker placing a frame only while independent
references agree on where it goes, and never repeating the last accepted pose as
though tracking had continued across evidence it did not have. Replaying the
first 80 of the capture's 176 frames — `--frames 80`, the run this section is
written from — it offers 71 and stops: 66 accepted (1 anchor, 15 adjacent, 43
model, 7 bounded coarse recoveries, one of the model placements a supported
model-only rescue), 5 refused, the first at `frame_0009`,
terminal loss at `frame_0070`, and the 9 frames after it listed but never
offered.

**The task-023 boundary is gone.** That run reached a terminal loss at
`frame_0034` after three candidate-disagreement refusals at 32, 33 and 34, and
`frame_0035` was never offered. All four are now placed by the ORDINARY
references — 32 and 33 against the model, 34 and 35 against the adjacent pair —
and the trajectory continues past them. The frame that bought that continuation
is `frame_0029`:

| | `frame_0029` |
| --- | --- |
| why the ordinary decision refused | three references placed it and disagreed |
| partitions | accepted frames 19, 21, 23, 25 and 20, 22, 24, 26 — disjoint |
| measured viewpoint span of each | 13.64 and 13.85 keyframe motion gates |
| coarse ranking | the last accepted pose, offered as a prior, covered the reference 0.456; the best genuinely different orientation 0.103 |
| coverage margin, on the selected hypothesis | 4.425x, against 1.15 |
| `coarse_align`'s own margin | 1.943x, against its own 1.15 |
| independent confirmation | the second partition placed it 0.567 from where the first did, against the unchanged 5.200 agreement gate |
| cost | 1 coarse search, 24 poses ranked, 4 hypotheses refined, 9 ICP solves — the policy ceiling |

Six more frames were recovered the same way — 41, 42, 46, 47, 61 and 62 — and
each kept the chain alive for the ordinary references that followed. Frame 46
is the direct regression for treating a principal-frame warning as a veto: its
reference is flagged near-degenerate, while the measured score margin
(`2.151x`), coverage margin (`3.083x`) and disjoint confirmation disagreement
(`1.769`) all clear their unchanged gates.

Four recovery attempts refused, and the reasons are the vocabulary working.
`frame_0009` has an actually ambiguous shared ranking (`1.004x` against the
`1.15x` margin). Frame 63's ranking is clear, but none of its four refined
hypotheses clears the ordinary gates; the refusal is isolated and frames 64–67
place normally. Frames 68 and 69 again have ambiguous shared rankings
(`1.026x` and `1.029x`). `frame_0070` is declined before any search: every
reference that aligned it at all called it a jump, and a better starting pose
cannot move the last accepted pose the travel gate measures against.

**Why it stops now, and what the boundary frames actually show.** The terminal
loss at `frame_0070` is three consecutive `tracking_no_candidate` refusals, each
one measured rather than guessed:

- `frame_0068`: the adjacent residual is 2.727 against the unchanged 2.600 gate,
  the model placement overlaps its reference by only 0.06, and the keyframe
  solve finds 0 of 395 correspondences.
- `frame_0069`: the model placement overlaps its reference by only 0.09, and
  the other two solves find 0 and 3 of 580 correspondences.
- `frame_0070`: the adjacent placement moves the surface 20.662 since the last
  accepted frame against the unchanged 15.600 travel gate, the keyframe finds
  0.28 of the frame on its reference against the 0.55 gate, and the model
  residual is 2.647.

The returns themselves are normal at the boundary: 688, 809 and 716 prepared
points at 68, 69 and 70 rather than empty or nearly empty clouds.
The boundary frames were inspected read-only, as occupancy maps of those returns
over a common window: a large structured surface still fills the view at 68, 69
and 70, and the silhouette MOVES sharply between 67 and 68. **The object had not
left the view.** So the honest limitation is not "the chassis went out of frame";
it is that between frames 67 and 70 the operator moved fast enough that no
reference could place the frame inside the unchanged residual and travel gates
while the chassis was still plainly visible. That is a capture-technique
boundary and a remaining product limitation, and it is recorded here as one
rather than described as a success. No accuracy claim is made for any accepted
pose on this capture: there is no surveyed truth for it, and the replay offers
none.

Over all 71 offered frames: 11 recovery attempts, 11 coarse searches, 1 recovery
declined before a search, 7 placements, and a worst case per recovery of 9 ICP
solves against a policy ceiling of 9, 25 orientations ranked against 26, 4
hypotheses refined against 4, 8 reference frames against 8 and 3,310 reference
points against 3,600 — every count inside the bound the policy dump states.
The worst offered frame took 9.47 seconds against a 1.10-second median, and the
whole 71-frame replay took 114.4 seconds. Recovery is paid for only on a refusal
or on the bounded tied-consensus arbitration; an unambiguous ordinary placement
still runs no coarse search at all.

## What this does NOT do

- It does not establish that an accepted pose is correct. It establishes that
  independent bounded references agreed on it under named thresholds. Every one
  of those references descends from earlier accepted poses, so a systematic
  error established early is agreed with rather than detected.
- It does not find where a repositioned object went. That is
  `src/scanner3d/relocalize.py`. The one search this module runs is the bounded
  coarse recovery below, and it is a re-seed of a refusal against the accepted
  model, not a search for an object that moved.
- It does not close loops or optimise a pose graph. `src/scanner3d/posegraph.py`
  runs afterwards, over the chain this produces — and on a well-tracked sweep it
  may now be *declined*, because `reconstruct_sweep` judges it against held-out
  revisits and returns whichever of the two chains agrees with them better.
- It does not remove backgrounds, hands or support surfaces. That is
  `src/scanner3d/preparation.py`.
- It is not bit-reproducible across compute backends. Keyframe selection is a
  threshold on a continuous measurement, so two runs whose poses differ in the
  fifth decimal can land on opposite sides of it, hold different keyframe sets,
  build different model references from them, and end with poses that differ by
  more than arithmetic would. That is not hypothetical: on the reconstruction
  suite's 24-frame looping fixture the python and scipy backends agree on every
  per-frame decision through frame 14 and then retain a different keyframe from
  frame 15 — the first frame a lap-closing revisit can seed from. Away from that
  threshold (the same fixture's first 15 frames) they agree exactly: same
  origin, same rescue flag and same keyframe set for every frame, with poses
  within 6.1e-4 of rotation-matrix entry and 0.0505 depth units of translation.
  `tests/test_pipeline.py::test_backend_does_not_change_the_answer` asserts the
  per-frame decisions rather than an aggregate count, so a backend that decided
  something different would fail rather than average out.
- Nothing here has been validated against a physical scan of known dimensions.
  The synthetic numbers above are a relationship measured under conditions where
  both chains were given exactly the same data; they are not accuracy.

## Two hand-held captures, 2026-09-04

Both were made by the operator with the desktop application at what were then
its defaults: one frame kept in three, spacing not held. The scanner delivers
about 4.5 depth frames a second, so the tracker saw frames 0.67 s apart. The
motion column is measured independently of the tracker, by a wide-reach ICP
between consecutive kept frames (scratch script `pairmotion.py`).

| capture | frames | live outcome | replay with the prediction-judged gate | motion between kept frames |
| --- | --- | --- | --- | --- |
| 20260904-120924 | 44 | lost at frame 13 on the old travel gate (19.9 and 25.9 mm against 15.6) | 42 of 44 placed; 2 refused for candidate disagreement (10.9 and 12.3 against the 5.2 gate) and recovered | 4-13 mm of surface travel |
| 20260904-164506 | 27 | lost at frame 24: adjacent overlap 0.32 under the 0.55 gate | the same: 24 of 27, no reference places 24, 25 (travel 45.5 and 38.5) or 26 (model overlap 0.16) | 5-14 mm of surface travel, camera translation 6-36 mm and up to 11 deg per kept frame; 23 to 24 unfittable at 12, 25 and 40 mm reach |

The first failure was the gate and is fixed. The second is the spacing: at
0.67 s apart, a hand at 3-5 cm/s puts 20-36 mm and a turn between the frames
the tracker sees, and the frames either side of the loss do not overlap enough
for any registration to bridge them. Seen at every frame the same motion is
7-12 mm. Hence held spacing is now on by default (`docs/frame-spacing.md`).
Two other things the hand-held frames show. A hand-held view sees the room past
the cloth: frame 23 carries a band of alias returns across the top of the image
(`docs/phantom-aliases.md`), which the depth band clips out of the registration
and which no cloth behind a turntable can prevent. And the tracker's own
diagnosis was right both times; what it could not say is that the cure for the
second was the spacing control beside it.
