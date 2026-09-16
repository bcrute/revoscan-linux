"""Where each ordinary frame goes, decided against more than the frame before it.

Why this exists
---------------
Until this module, every ordinary frame of a capture was placed by one point-to-
plane solve against its immediate predecessor, and whatever that solve returned
became the pose. That is the right idea for a slowly turning object filmed by a
fixed scanner - two consecutive frames really are nearly the same picture - and
it is the wrong idea for the capture the owner actually wants to make.

Carrying the scanner around the mapper chassis breaks the assumption in two
different ways at once:

- **Individual views go weak.** Pointed at one flat panel, a single frame has no
  geometry that determines rotation about the panel's normal. Adjacent
  point-to-plane on that pair is singular in a direction; it does not report
  that, it reports a small residual for a transform that slid sideways.
- **Nothing corrects the chain.** A pose is the composition of every previous
  relative fit, so every one of those errors is permanent. There is no reference
  outside the pair, so a chain that drifts drifts in a way no measurement inside
  it can see.

On 2026-08-28 both of those showed up in one real capture: the model updated two
or three times and then sat still while the operator kept walking, and nothing
in the software said so. Support-plane suppression (task 022) was necessary and
did not fix it, because the problem was never the bed - it was that a pose came
from one pair and one pair only.

What this module does instead
-----------------------------
For each ordinary frame it forms up to :data:`MAX_CANDIDATES` INDEPENDENT pose
candidates and then decides between them:

1. the **adjacent** candidate, from the immediately previous accepted frame -
   the same solve the capture has always done, kept because it is the most
   direct measurement there is when it works;
2. a bounded number of **keyframe** candidates, from accepted frames chosen for
   the viewpoints they hold rather than for being recent;
3. a **model** candidate, from a bounded reference sampled across several
   accepted frames and expressed in accepted-model coordinates, so a frame that
   sees a surface built up over many earlier viewpoints is fitted against all of
   it at once instead of against one noisy view of it.

Every ordinary candidate is a REFINEMENT with an explicit initial transform.
The ordinary decision does not search: it never asks "where on the model might
this go", it asks "does this stated starting pose hold up". Global coarse search
over a repositioned segment already exists, in :mod:`scanner3d.relocalize`, and
is a different and much more expensive decision.

Where that starting pose comes from
-----------------------------------
The scanner is carried, so the best cheap statement about where a frame is, is
where the last one was PLUS how the scanner was moving. This module keeps one
constant-velocity estimate - the step between the last two accepted frames, and
nothing more - and spends it in two places:

- as the prediction the keyframe and model refinements are seeded from, which
  previously fell back to the previous pose whenever the adjacent solve had not
  been accepted. That fallback asserts the scanner did not move at all, on
  precisely the frames where it had already failed to be placed;
- as ONE bounded retry of the adjacent candidate, and only where the ordinary
  identity-seeded attempt produced no accepted placement.

The identity attempt stays first, and that is a constraint rather than a
preference: :func:`~scanner3d.pipeline.reconstruct_sweep` pre-fits every
adjacent pair across worker processes before any pose chain exists to
extrapolate from, and a tracker that seeded its first adjacent solve from a
prediction could not accept those pre-fits without making the offline path
sequential. See :meth:`IncrementalTracker._adjacent_candidate`.

The prediction changes no gate. It decides which local minimum a refinement can
reach and nothing else, so a badly predicted frame is refused by the same
residual, overlap, travel and agreement evidence as any other, and a frame with
no usable prediction is placed exactly as it was before this existed. The seed
each candidate was reached from is recorded on it - see
:data:`IDENTITY_SEED` and its neighbours - because "the previous frame placed
this" and "the previous frame placed this once it was told where to look" are
different statements about a capture.

When every one of those refinements is a refinement of the SAME stale
prediction, they can each settle into a different local minimum while the object
is still plainly in view - which is not a frame that cannot be placed, it is
three answers to a question nobody asked from a fresh starting pose. Measured on
the owner's 2026-08-28 capture: frames 32 to 34 refused for candidate
disagreement at 0.35 to 0.49 model overlap, with a substantial detailed chassis
visible throughout. So AFTER the ordinary decision has refused, or before it
accepts the one bounded case where tied consensus groups leave a distinctly
tighter fallback, one deterministic coarse-to-fine arbitration runs - see
:meth:`IncrementalTracker._maybe_recover`. It is not a second orientation
search: it calls the one this repository already has,
:func:`~scanner3d.multiview.coarse_align`, offers it the last accepted pose as
one more initial pose to rank, ranks everything once, refuses unless that
ranking is unambiguous by the shared verdict AND by coverage, refines the best
few, re-measures the coverage margin from the hypothesis it actually selected,
and accepts a placement only when two reference sets that share no accepted
frame and each span real viewpoint diversity both clear every ordinary gate on
it and agree. A frame with one unambiguous ordinary placement never reaches any
of that and pays none of it. When the tied-cluster fallback reaches it, a failed
recovery leaves that fallback intact and attaches the failed evidence to the
accepted decision rather than turning uncertainty about stronger evidence into
a tracking loss.

Measured on that capture with this in place: frame 29's references disagreed,
the recovery ranked the last accepted pose 4.4x clear of the best genuinely
different orientation, two disjoint partitions placed it 0.567 apart against the
unchanged 5.200 agreement gate, and frames 30 to 39 - including 32 to 34 - were
then placed by the ORDINARY references. The recovery's value there is one frame;
what that one frame bought was the rest of the capture.

What "accepted" is allowed to mean
----------------------------------
Not "ICP returned". Every candidate carries the four measurements the rest of
this repository already judges an alignment by - finiteness, convergence,
residual against the measured noise floor, matched and used fractions - judged
by the SAME function the tentative local chain uses
(:func:`scanner3d.relocalize.judge_local_edge`), plus two more this decision
needs:

- **overlap** with the reference it was fitted against, because a small residual
  over a corner of the frame is not evidence about the frame; and
- **travel**, the distance the surface moved since the last accepted pose,
  because a placement that teleports the scanner is refused whatever its
  residual says.

Then the candidates are compared, and this is the part that matters most:

- If two or more independent candidates survive their gates and AGREE, that
  agreement is the evidence, and the pose follows :data:`PREFERENCE_ORDER`:
  model, then adjacent, then keyframe. The model provides broader accumulated
  geometry; the adjacent refinement precedes a keyframe to avoid repeatedly
  paying a viewpoint-dependent keyframe offset (see the origin definitions).
  Candidate agreement is not an independent guarantee against long-run drift.
- If the largest mutual-agreement groups tie, the frame is refused unless one
  group is distinctly tighter than the next by the shared ambiguity margin.
  Tightness is the widest pose disagreement inside the group, never an ICP
  residual: two equally plausible placements remain two answers. A distinctly
  tighter pair is a bounded fallback, but the disjoint multi-frame recovery is
  stronger evidence and gets the first chance to place that frame; the pair is
  used only if the stronger arbitration cannot confirm a replacement.
- If only the model survives - the adjacent pair was singular, which is exactly
  the case this module exists for - the placement is a RESCUE and is held to
  stricter gates: the reference must span several accepted viewpoints and the
  frame must overlap it well. A single planar view rescued off a reference that
  is itself one viewpoint is not evidence, it is the same singularity wearing a
  different name.

A refusal changes nothing
-------------------------
A refused frame does not append a pose, does not touch the keyframe set, does
not slide the retained clouds and does not move the model reference. That is
load-bearing rather than tidy: the caller publishes a failure streak from these
refusals, and a tracker that had half-mutated itself would make the next frame's
evidence describe a state no decision ever accepted.

What this does NOT do
---------------------
It does not find where a repositioned object went - see
:mod:`scanner3d.relocalize`; the coarse fallback above re-seeds a refinement
inside the same travel gate every ordinary placement clears, which is a much
smaller claim. It does not close loops or optimise a pose graph -
see :mod:`scanner3d.posegraph`, which runs afterwards over the chain this
produces. It does not remove backgrounds, hands or support surfaces - see
:mod:`scanner3d.preparation`. And it does not establish that an accepted pose is
CORRECT: it establishes that several independent references agreed on it under
named thresholds, which is a much stronger statement than the chain it replaces
could make and still not a measurement of truth.

No physical unit is implied. Distances are in whatever the depth scale produced,
and every threshold is expressed as a multiple of
:data:`scanner3d.pipeline.NOISE_FLOOR` so it stays comparable with every other
alignment threshold here.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid, spanned_dimensions
from scanner3d.icp_plane import icp_point_to_plane
from scanner3d.multiview import (
    AMBIGUITY_MARGIN,
    AlignmentCandidate,
    CoarseAlignment,
    coarse_align,
    octahedral_rotations,
    overlap_fraction,
)
from scanner3d.pipeline import NOISE_FLOOR, frame_travel
from scanner3d.registration import DEFAULT_TOLERANCE, cloud_extent, icp
from scanner3d.relocalize import (
    LOCAL_RESIDUAL_MULTIPLE,
    OVERLAP_TOLERANCE_MULTIPLE,
    LocalEdge,
    LocalFit,
    Refusals,
    judge_local_edge,
    local_chain_gates,
)

SCHEMA_VERSION: Final = 6
"""5 lets a disjoint-reference recovery arbitrate a clearly tighter tied cluster first.

Version 4 made principal-frame degeneracy evidence rather than a recovery veto.
Version 3 added the ordinary consensus-cluster ambiguity margin to the policy
record. Version 2 added the ``recovery`` key to :meth:`TrackDecision.as_dict`
and the recovery bounds to :meth:`TrackingPolicy.as_dict`. Nothing in the
earlier versions changed spelling: a reader of a version-1 record is reading a
tracker that could not attempt a bounded coarse recovery; version 2 refused
every tie between maximum agreement clusters without measuring their tightness;
version 3 accepted a distinctly tighter tied cluster before offering the
stronger disjoint-reference recovery; and version 4 still refused a recovery
whenever either principal frame had near-equal extents, even if both ambiguity
verdicts and an independent confirmation distinguished the placement.
"""

#: Version of the bounded coarse recovery's own evidence record, apart from
#: :data:`SCHEMA_VERSION` so that a reader comparing a live capture's recovery
#: with an offline reconstruction's is comparing a stated thing rather than an
#: assumed one.
RECOVERY_SCHEMA_VERSION: Final = 2
"""2 records recoveries where a principal-frame warning no longer acts as a veto.

Version 1 always refused before refinement when either coarse principal frame
had near-equal extents. Version 2 retains both degeneracy flags and the shared
coarse limitations but lets the measured ambiguity, placement and independent
confirmation gates below judge the recovery.
"""

#: Fewest points a frame must carry before a placement is even attempted.
#:
#: Three: the fewest that can span a plane, and therefore the fewest that can be
#: the input to any alignment at all. Deliberately NOT a large number dressed up
#: as an object-left-view test. A scanner pointed past the chassis at nothing in
#: range does not return zero points, it returns a few hundred noise points, and
#: a count threshold that tried to catch that would have to be tuned to one
#: scene and would refuse legitimate thin views everywhere else. What actually
#: catches the object leaving the view is downstream: no candidate can be formed
#: from a scatter (ICP refuses it), and any that is formed fails the residual,
#: matched, used and overlap gates. This bound exists so an EMPTY frame is
#: refused before anything is asked to align it.
MIN_TRACK_POINTS: Final = 3

#: Accepted frames kept as keyframes. The set is a bounded description of the
#: viewpoints the capture has visited, not an archive of it: past this the most
#: redundant non-anchor, non-recent keyframe is evicted, so a two-hour capture
#: costs exactly what a two-minute one does.
MAX_KEYFRAMES: Final = 12

#: Exact immutable pose-pair results, not an additional reference archive.
#: A 64-keyframe diagnostic policy evaluates over 4,000 directed pairs; the
#: old 2,048-entry cache thrashed on each eviction pass. Leave bounded room
#: for those pairs plus current-frame selection without changing arithmetic.
POSE_GAP_CACHE_PAIRS: Final = 16_384

#: Keyframes refined against as candidates for one frame. One, deliberately:
#: each candidate is a full ICP solve, the model candidate below is already
#: built FROM the keyframe set, and this is the per-frame cost that a long
#: capture pays on every single frame. The keyframe chosen is the one nearest
#: the predicted pose, which is the one most likely to actually overlap.
KEYFRAME_CANDIDATES: Final = 1

#: Newest keyframes that may never be evicted, so the set always holds recent
#: coverage as well as the anchor and the spread between them. Without this an
#: eviction rule that drops the most redundant entry would eventually drop the
#: viewpoint the scanner is standing in.
PROTECTED_RECENT_KEYFRAMES: Final = 2

#: Most accepted clouds inspected while building one model reference. The hard
#: bound on "how much of the model does a frame get compared against", and the
#: reason the cost of placing frame 900 equals the cost of placing frame 9.
MAX_REFERENCE_FRAMES: Final = 8

#: Points in the bounded model reference, spread across the frames selected for
#: it. Roughly twice a live preview frame's own 900 points, because the
#: reference has to be a SURFACE at the density the frame is, not a sketch of
#: one: measured on this repository's four-frame bridged arc, a 600-point
#: reference put the model candidate 5.2 degrees from the keyframe candidate,
#: and an 1800-point one put it 1.2 degrees away - inside the agreement band
#: rather than outside it. A reference too sparse to agree with anything turns
#: the disagreement gate into a refusal generator.
MODEL_REFERENCE_POINTS: Final = 1800

#: Points of the offered frame used as the source of EVERY candidate, the
#: adjacent one included.
#:
#: Round 1 of this task exempted the adjacent candidate on the grounds that it
#: was "the solve the capture has always done", and the physical replay showed
#: what that exemption actually bought: a frame carrying about 18000 points was
#: fitted, travelled and retained at 18000 points against a nominal 900-point
#: budget, and 176 frames took 738.971 seconds. A bound that any candidate may
#: opt out of is not a bound. This is the live preview's own frame size, so on a
#: live capture the sample is the whole frame and every candidate is fitted from
#: identical geometry; on a denser offline cloud it is the ceiling, and the
#: offline path samples the pairs it pre-fits the SAME way (see
#: :func:`bounded_source`) so the two paths still solve the identical problem.
CANDIDATE_SOURCE_POINTS: Final = 900

#: Points kept per accepted cloud. Every reference this module builds is
#: assembled from retained clouds, so this is what makes "reference points" and
#: "memory" bounds rather than descriptions: an accepted frame costs at most
#: this many points however many the caller offered.
#:
#: Equal to :data:`CANDIDATE_SOURCE_POINTS` deliberately. The previous accepted
#: cloud is the adjacent candidate's REFERENCE and the offered frame is its
#: SOURCE, and a reference thinner than the source it is matched against lowers
#: the matched fraction for a reason that has nothing to do with the geometry.
RETAINED_CLOUD_POINTS: Final = CANDIDATE_SOURCE_POINTS

#: How much of the noise floor a solver iteration must stop moving the residual
#: by before the fit counts as settled.
#:
#: This replaces the round-1 arrangement, in which a non-converged fit was
#: rewritten as a converged one for gating. That was the wrong repair for a real
#: observation. The observation was sound:
#: :func:`~scanner3d.icp_plane.icp_point_to_plane` measures convergence against
#: ``tolerance`` times the TARGET'S BOUNDING-BOX DIAGONAL, and its 1e-6 default
#: is about 6e-5 on a 66-unit object - a thousandth of a thousandth of the
#: sensor's own noise, so a healthy fit whose trimmed correspondence set
#: reshuffles slightly every pass never meets it and runs to the iteration cap.
#: Measured on this repository's own 24-frame sweep at the 1e-6 default:
#: residuals of 0.06 to 0.18, matched and used fractions at their gates, and
#: frames that still reported ``converged=False`` at 50 iterations.
#:
#: The repair is to state the threshold in the units the decision is actually
#: made in. An ICP residual is a distance in depth units; the smallest distance
#: that means anything on this sensor is :data:`~scanner3d.pipeline.NOISE_FLOOR`;
#: so a residual that stopped improving by as much as a HUNDREDTH of the noise
#: floor has stopped improving. That threshold is 0.0065 depth units, it is the
#: same absolute threshold whatever the object's extent, it is recorded in the
#: policy, and it is deterministic. Measured on the same sweep it converges
#: every frame in 5 to 9 iterations at residuals matching the 1e-6 run to three
#: decimals - the same fits, honestly reported as settled - and a fit that
#: genuinely never settles is now refused by
#: :func:`~scanner3d.relocalize.judge_local_edge` like any other failed gate.
CONVERGENCE_NOISE_FRACTION: Final = 0.01

#: Accepted clouds held in memory. Everything the decision needs is either a
#: keyframe or recent, so this bounds the tracker's memory without bounding the
#: capture: poses are kept for every accepted frame, clouds are not.
MAX_RETAINED_CLOUDS: Final = 32

#: Accepted viewpoints a model reference must span before a model candidate is
#: formed at all. Two: below that the reference is one accepted frame, which is
#: what the keyframe candidate already is, and two names for one measurement do
#: not agree with each other - they repeat each other, at the preference the
#: broader reference is supposed to have earned.
MIN_MODEL_REFERENCE_FRAMES: Final = 2

#: Most recent accepted frames that join the keyframes in the model reference,
#: excluding the immediately previous one - see
#: :meth:`IncrementalTracker._reference_frames` for why that exclusion is what
#: makes the model candidate independent evidence rather than a second opinion
#: from the same pair.
MODEL_RECENT_FRAMES: Final = 2

#: The hard ceiling on ICP solves this module will run for one offered frame:
#: one adjacent, :data:`KEYFRAME_CANDIDATES` keyframes, one model.
MAX_CANDIDATES: Final = 1 + KEYFRAME_CANDIDATES + 1

#: Fraction of a frame that must land on a KEYFRAME OR MODEL reference for its
#: residual to be describing the frame rather than a corner of it.
#:
#: Not applied to the adjacent candidate, and that is a stated choice rather
#: than an oversight: for two consecutive frames the reference IS the previous
#: cloud at full density, so
#: :attr:`~scanner3d.relocalize.LocalEdge.matched_fraction` already measures
#: exactly this and is already gated by the local-edge vocabulary. Overlap is
#: still measured and recorded for the adjacent candidate, because a reader
#: comparing candidates needs the same numbers for all of them.
MIN_REFERENCE_OVERLAP: Final = 0.35

#: Overlap a MODEL-ONLY rescue must clear. Higher than
#: :data:`MIN_REFERENCE_OVERLAP` because a rescue has no second opinion: the
#: adjacent pair was singular or refused, so the model reference is the only
#: thing saying where this frame is, and the frame has to be substantially ON
#: that reference before that is worth believing.
RESCUE_MIN_OVERLAP: Final = 0.55

#: Accepted frames the model reference must be built from before a MODEL-ONLY
#: rescue is allowed. Three, because the failure this rescues from is a view
#: with no geometry determining one direction, and a reference assembled from
#: one or two nearby viewpoints has the same blind direction as the frame does.
#: Several viewpoints see that direction from elsewhere; that is the whole
#: reason the rescue can work at all.
RESCUE_MIN_REFERENCE_FRAMES: Final = 3

#: How far the surface may move between two accepted frames, as a multiple of
#: the noise floor, before the placement is refused as implausible.
#:
#: Deliberately far above :data:`~scanner3d.pipeline.TRAVEL_MAX_MULTIPLE`, which
#: is CAPTURE GUIDANCE - "you are turning too fast for good results" - and not a
#: refusal. This is the other thing entirely: the distance beyond which a
#: placement is not a fast scan but a jump to somewhere else on the object. A
#: gate set at the guidance value would refuse ordinary brisk captures.
MAX_TRAVEL_MULTIPLE: Final = 24.0

#: Delivery steps of the last accepted motion one prediction may be carried
#: forward over.
#:
#: This module numbers frames by ACCEPTANCE, so a refused frame is invisible to
#: everything downstream of it: the next frame is two delivery steps from the
#: reference it is seeded against, the one after that three. That is the
#: mechanism behind docs/tracking.md's "one refused frame leaves it a whole
#: skipped step away from the answer", and composing the last accepted step
#: that many times is the body-frame constant-velocity prediction of where the
#: frame should now be.
#:
#: Three, because a prediction is an extrapolation and every step of it
#: multiplies whatever the previous one got wrong. Past three the scanner has
#: been unplaceable for the better part of a second at the measured 4.5 fps,
#: which is the tracking-loss streak's own territory - and a prior confident
#: enough to seed a solve from is precisely what must not be manufactured
#: there.
MAX_PRIOR_STEPS: Final = 3

#: How far the prediction itself may carry the surface, as a multiple of the
#: travel gate, before it is a runaway extrapolation rather than a velocity.
#:
#: A prediction used to be discarded the moment it crossed the travel gate,
#: because a fit near it would have been refused as a jump anyway. That is no
#: longer true: the travel gate is judged against the prediction where one
#: exists (see :meth:`IncrementalTracker._judge`), which is what lets a
#: scanner carried steadily at a speed the gate alone would call a jump keep
#: being placed. The cap stays, because an extrapolation compounds whatever
#: the last step got wrong: three gates is 46.8 depth units of surface travel
#: in one delivery step, about 21 cm/s at the measured 4.5 fps, past which a
#: fringe-projection frame is motion-corrupted before any tracker sees it.
MAX_PRIOR_TRAVEL_MULTIPLE: Final = 3.0

#: How far apart two candidate placements may put the SURFACE before they are
#: two answers rather than one.
#:
#: Measured the way this module measures everything else about a placement: the
#: median distance the frame's own points move between the two poses, from
#: :func:`~scanner3d.pipeline.frame_travel`, in depth units and expressed as a
#: multiple of the noise floor. Not a rotation angle and a translation compared
#: separately, because those two cannot be added and because neither says what
#: matters - a two-degree rotation about an axis through the object is nothing,
#: and the same two degrees about the sensor is centimetres.
#:
#: The multiple is DERIVED rather than chosen. The residual gate accepts a fit
#: whose points sit up to :data:`~scanner3d.relocalize.LOCAL_RESIDUAL_MULTIPLE`
#: noise floors off the reference. Two placements that each satisfy that can be
#: up to twice that far from EACH OTHER while both remaining consistent with the
#: data they were fitted to, so anything inside twice the residual gate is not
#: evidence of a contradiction; anything beyond it is.
AGREEMENT_RESIDUAL_MULTIPLE: Final = 2.0 * LOCAL_RESIDUAL_MULTIPLE

#: How much tighter one equally large mutual-agreement cluster must be than the
#: next before it is one answer with a dissenting reference rather than two
#: answers.
#:
#: Imported rather than chosen: :data:`~scanner3d.multiview.AMBIGUITY_MARGIN`
#: is already this repository's boundary for "the evidence distinguishes the
#: best from the runner-up". Here smaller is better, so the margin is
#: ``runner_up_diameter / best_diameter``. The diameter is the widest pairwise
#: surface displacement inside a cluster, measured by the same
#: :func:`_disagreement` and in the same units as the agreement gate itself.
#: No ICP residual is used to arbitrate between placements.
#:
#: This matters at a chain of three candidates. A--B and B--C may each sit
#: inside the agreement gate while A--C does not. If the two pair diameters are
#: comparable, the old and still-correct answer is "two placements". If one
#: pair is distinctly tighter, that pair is a bounded fallback and the third
#: reference is its dissenter. A disjoint multi-frame recovery is tried first
#: because it is stronger evidence. Measured on ``20260831-175352`` frame 88:
#: adjacent/keyframe were 1.620 apart and adjacent/model 4.760 apart, a 2.94x
#: margin; recovery could not confirm a replacement by 0.01 of overlap, the
#: fallback placed the frame, and all three references agreed again afterwards.
AGREEMENT_CLUSTER_MARGIN: Final = AMBIGUITY_MARGIN

#: Rotation and distance below which the scanner counts as not having moved.
#: A keyframe is never taken from a frame inside this band, which is what stops
#: a scanner left pointing at one spot from filling the keyframe set with twelve
#: copies of one view.
STATIONARY_DEGREES: Final = 1.0
STATIONARY_DISTANCE_MULTIPLE: Final = 1.0

#: How far an accepted pose must sit from EVERY existing keyframe before it is
#: itself worth keeping as one. This is the motion half of keyframe selection.
KEYFRAME_MIN_DEGREES: Final = 6.0
KEYFRAME_MIN_DISTANCE_MULTIPLE: Final = 6.0

#: Overlap below which an accepted frame counts as showing something new, even
#: though it has not moved far enough to clear the motion gate above. This is
#: the novelty half: rounding a corner moves the scanner very little and changes
#: what it can see entirely, and a keyframe set chosen on motion alone misses
#: exactly that.
KEYFRAME_NOVEL_OVERLAP: Final = 0.60

#: What the caller records as the failure type when a frame is refused. Free
#: strings rather than an enum for the same reason
#: :data:`~scanner3d.relocalize.Refusals` is: the durable record carries them
#: through to a reader who has only the file.
TRACK_FRAME_EMPTY_TYPE: Final = "tracking_frame_empty"
TRACK_FRAME_UNUSABLE_TYPE: Final = "tracking_frame_unusable"
TRACK_NO_CANDIDATE_TYPE: Final = "tracking_no_candidate"
TRACK_DISAGREEMENT_TYPE: Final = "tracking_candidates_disagree"
TRACK_RESCUE_UNSUPPORTED_TYPE: Final = "tracking_model_rescue_unsupported"

#: Origins, in the order this module prefers them when candidates agree.
#:
#: The model first, because it is the only reference here that is BROADER than
#: one view: it is sampled across several accepted viewpoints in model
#: coordinates, so a pose measured against it is not a composition of pairwise
#: errors and does not accumulate them. That is the whole mechanism - measured
#: on the moving-camera sweep in ``tests/test_tracking.py``, preferring it holds
#: the error flat where every other choice lets it climb like a chain's.
#:
#: Then the adjacent pair, and the keyframe LAST, which looks backwards and is
#: not. A keyframe and the previous frame are both exactly one accepted
#: viewpoint; neither is broader than the other, so the "broader reference wins"
#: argument says nothing about the pair of them. What does say something is
#: which of the two the frame overlaps more, and that is the immediately
#: previous one by construction. Measured: preferring a keyframe over the
#: adjacent pair on the third frame of a sweep - where the only keyframe is the
#: anchor, a quarter of a turn away - placed it 1.2 degrees out where the
#: adjacent pair placed it 0.2 out.
#:
#: When they agree the disagreement gate has already done its work, so this
#: order is not deciding WHICH answer; it is deciding which measurement of the
#: same answer to keep.
#: How much of the tracker's OWN correspondence distance the first of the two
#: recovery refinement passes runs at.
#:
#: This is the "coarse-to-fine" in the name, and it is a SEED schedule rather
#: than a gate. A coarse hypothesis from :func:`~scanner3d.multiview.coarse_align`
#: is a whole-object orientation ranked on a decimated cloud over a coarse grid;
#: measured on this repository's asymmetric recovery fixture the best one sits
#: 33 to 40 degrees from the truth. Point-to-plane ICP started there cannot walk
#: that far at an ordinary correspondence distance - the correspondences it needs
#: are outside the gate before the first iteration - so the first pass runs at
#: four times that distance to get into the basin, and the SECOND pass, the one
#: whose fit is judged, runs at the tracker's ordinary distance from the first
#: pass's answer. Measured on that fixture: 35.4 degrees coarse, 0.11 degrees
#: after the pair, and every wrong hypothesis still refused by travel and
#: overlap. No gate is widened by this: the wide pass produces a starting pose
#: and nothing else, and the fit that reaches
#: :func:`~scanner3d.relocalize.judge_local_edge` is the ordinary one.
RECOVERY_COARSE_DISTANCE_MULTIPLE: Final = 4.0

#: Points the coarse RANKING is scored from.
#:
#: Not sampled from the offered frame but from the bounded candidate source, so
#: it is a SUBSET of the sample every ordinary candidate was already fitted
#: from: the recovery ranks the geometry the ordinary decision failed on rather
#: than a denser cloud nothing else saw. That also makes this bound
#: automatically no looser than :data:`CANDIDATE_SOURCE_POINTS` whatever a
#: policy sets it to, which is why there is no relationship check for the pair.
#: Bounded separately from the refinement because the ranking scores every
#: hypothesis and a refinement scores one.
RECOVERY_SOURCE_POINTS: Final = 600

#: Disjoint recovery reference partitions built for one offered frame.
#:
#: Two: one to search and refine against, one to CONFIRM against, and they share
#: no accepted frame. One partition cannot confirm itself, and a third would buy
#: a second confirmation at the price of thinning both of the first two - the
#: accepted frames a capture has are the budget here, not the arithmetic.
RECOVERY_PARTITIONS: Final = 2

#: Accepted frames inspected per partition. With
#: :data:`RECOVERY_PARTITIONS` this is the hard ceiling on accepted frames one
#: recovery may look at: eight, however long the capture has run.
RECOVERY_PARTITION_FRAMES: Final = 4

#: Accepted viewpoints each partition must span before it may confirm anything.
#: Two, the same number :data:`MIN_MODEL_REFERENCE_FRAMES` requires of a model
#: reference and for the same reason: one accepted frame is one viewpoint, and a
#: placement confirmed by one viewpoint is blind in whatever direction that
#: viewpoint is blind in.
RECOVERY_MIN_PARTITION_FRAMES: Final = 2

#: How far apart, in KEYFRAME MOTION GATES, the two furthest-separated frames of
#: one recovery partition must sit before that partition may confirm anything.
#:
#: Distinct frame indices are not distinct viewpoints. A scanner held still for
#: a second produces accepted frames 21, 23, 25 and 27 that are four readings of
#: ONE view, and a partition assembled from them is blind in exactly the
#: direction the offered frame is - the case the rescue frame-count gate exists
#: to refuse, arriving by a route that gate cannot see because it counts frames.
#: So the span is MEASURED, in :meth:`IncrementalTracker._pose_distance`, which
#: is the tracker's own already-tested pose-gap vocabulary: degrees over
#: :data:`KEYFRAME_MIN_DEGREES` and distance over
#: :data:`KEYFRAME_MIN_DISTANCE_MULTIPLE` times the noise floor, whichever is
#: larger. One gate is the threshold because that is the same displacement the
#: tracker itself calls a new viewpoint worth keeping a keyframe for; a
#: partition that does not reach it has not moved far enough for its own frames
#: to disagree with each other.
RECOVERY_MIN_PARTITION_POSE_SPAN: Final = 1.0

#: Points in each partition: :data:`MODEL_REFERENCE_POINTS`, the same size and
#: for the same reason as the model reference a recovery partition stands in
#: for.
#:
#: It has to be that size, and the physical replay is what says so. Every
#: hypothesis a recovery refines is measured by the SAME overlap gate an
#: ordinary model placement is measured by, so a partition thinner than the
#: model reference makes that gate measure the reference's sparsity rather than
#: the frame's placement. Measured on ``20260828-105933`` at ``frame_0032``,
#: with this at 900: the ordinary model candidate overlapped its 1800-point
#: reference 0.47 and the recovery's best hypothesis overlapped its 796-point
#: partition 0.19, against the same 0.35 gate - so the recovery refused a
#: placement the ordinary vocabulary would have let through, for a reason that
#: was about its own budget and not about the frame.
RECOVERY_PARTITION_POINTS: Final = MODEL_REFERENCE_POINTS

#: Initial poses this module OFFERS to the shared coarse search on top of the
#: orientations that search generates for itself: exactly one, the last accepted
#: pose.
#:
#: A recovery is reached for a frame whose references disagreed, not for a frame
#: that is lost, and the last accepted pose is the one placement in the capture
#: this module already trusts - it is the pose the travel gate measures every
#: recovered placement against. Offering it to the RANKING rather than assuming
#: it is the whole point: it is scored by the same decimated pass on the same
#: grid as the generated orientations, wins or loses the ranking on that
#: measurement, and is refused by every gate the others are refused by. It buys
#: no confidence; what it buys is a starting pose whose centroid is where the
#: frame actually is. Measured on ``20260828-105933`` at ``frame_0029``: the
#: best generated orientation covered the recovery reference 0.10, because
#: :func:`~scanner3d.multiview.coarse_align` aligns a PARTIAL view's centroid
#: onto a multi-frame reference's centroid and a partial view's centroid is not
#: there; the last accepted pose covered it 0.46 and won the ranking 4.4x.
#:
#: It is not a second search, and this module still contains no orientation
#: search of its own.
RECOVERY_PRIORS: Final = 1

#: Orientations one call to :func:`~scanner3d.multiview.coarse_align` can rank:
#: the principal candidate plus the 24 octahedral rotations of it. DERIVED from
#: that function rather than written down beside it, so the ceiling this module
#: advertises cannot drift away from the number the search actually produces.
COARSE_ORIENTATIONS: Final = 1 + len(octahedral_rotations())

#: Everything one coarse ranking may hold: the generated orientations plus
#: :data:`RECOVERY_PRIORS`. This is the number a work count is bounded by.
COARSE_RANKED_POSES: Final = COARSE_ORIENTATIONS + RECOVERY_PRIORS

#: Coarse hypotheses this module SCORES FOR COVERAGE, best coarse score first,
#: before any of them is refined.
#:
#: A shortlist bound, and deliberately not called "the hypotheses the coarse
#: search ranked". The search ranks everything it can generate - up to
#: :data:`COARSE_ORIENTATIONS` of them - whatever this is set to, so a record
#: that reported this number as the work done would understate it by however
#: much a policy narrowed the shortlist. :class:`RecoveryWork` carries the two
#: counts separately for exactly that reason.
RECOVERY_RANKED_HYPOTHESES: Final = 25

#: Coarse hypotheses REFINED, best-ranked first. Each costs two ICP solves, so
#: this is the number that sets the recovery's cost; four covers the principal
#: candidate and the three nearest flips of it, which is where every hypothesis
#: that survived a gate on this repository's fixtures came from.
RECOVERY_HYPOTHESES: Final = 4

#: How much better a coarse hypothesis's COVERAGE must be than the best
#: genuinely different placement standing beside it.
#:
#: Asked twice: once of the ranking's own best, before any refinement, so that
#: nothing a solver returns afterwards can talk the decision out of it; and
#: again of the hypothesis actually SELECTED, after refinement, so the margin on
#: the record belongs to the placement the record is about. Neither replaces
#: :func:`~scanner3d.multiview.coarse_align`'s own verdict, which is required
#: as well.
#:
#: :data:`~scanner3d.multiview.AMBIGUITY_MARGIN` itself, imported rather than
#: chosen, because it is the same question the coarse stage already answers:
#: does the shape distinguish these placements. What differs is the quantity it
#: is applied to, and the two quantities are measured with different
#: instruments over different amounts of the frame.
#: :func:`~scanner3d.multiview.coarse_align` ranks on a truncated mean
#: nearest-neighbour distance over a DECIMATED cloud on a grid whose cell is 4%
#: of the target extent. This module measures COVERAGE - what
#: :func:`~scanner3d.multiview.overlap_fraction` reports over the WHOLE frame at
#: the same tolerance :meth:`IncrementalTracker._judge` measures every ordinary
#: candidate's overlap at, which is the tolerance the recovered placement will
#: be judged by if it is admitted. Measured on this repository's
#: incompatible-minima fixture: the shared score separates the ranking 2.42x,
#: the coverage of the ranking's best placement 1.36x at 0.375 against 0.276,
#: and the SELECTED hypothesis's own coverage 3.13x; on a near-symmetric slab
#: the coverage margin is 1.02 - ambiguous, and refused.
RECOVERY_AMBIGUITY_MARGIN: Final = AMBIGUITY_MARGIN

ADJACENT_ORIGIN: Final = "adjacent"
KEYFRAME_ORIGIN: Final = "keyframe"
MODEL_ORIGIN: Final = "model"
ANCHOR_ORIGIN: Final = "anchor"
BRIDGE_ORIGIN: Final = "bridge"
#: The two origins a bounded coarse recovery produces. Deliberately NOT in
#: :data:`PREFERENCE_ORDER`: a recovery is never one of several references being
#: preferred between, it is a separate decision reached after the ordinary one
#: has refused or while a distinctly tighter tied cluster awaits stronger
#: arbitration, and it is accepted only when BOTH of these agree.
RECOVERY_ORIGIN: Final = "coarse_recovery"
RECOVERY_CONFIRM_ORIGIN: Final = "coarse_recovery_confirmation"
PREFERENCE_ORDER: Final = (MODEL_ORIGIN, ADJACENT_ORIGIN, KEYFRAME_ORIGIN)

#: What a candidate's refinement was STARTED from, recorded beside the placement
#: it produced and never used as a gate. A seed decides which local minimum a
#: solve can fall into, so "the previous frame placed this" and "the previous
#: frame placed this once it was told roughly where to look" are two different
#: statements about how the capture is going, and a reader comparing two
#: captures needs to be able to tell them apart.
IDENTITY_SEED: Final = "identity"
PRIOR_SEED: Final = "motion_prior"
PREDICTED_SEED: Final = "predicted"
COARSE_SEED: Final = "coarse_search"
RECOVERED_SEED: Final = "recovered_placement"

#: The two gates this decision adds on top of the shared alignment vocabulary,
#: named so a caller can tell WHICH of them a candidate failed without parsing a
#: sentence back out of :attr:`TrackCandidate.refusals`. The recovery trigger
#: reads them: a frame every reference agrees is a JUMP has already been proved
#: implausible, and no starting pose can change where the last accepted frame is.
OVERLAP_GATE: Final = "overlap"
TRAVEL_GATE: Final = "travel"

RECOVERY_LIMITATIONS: Final = (
    "A bounded coarse recovery improves the STARTING POSE a refinement is run from. It does "
    "not widen a gate, and a placement it finds still has to clear the same convergence, "
    "residual, matched, used, overlap and travel evidence every ordinary placement clears.",
    "When equally large ordinary consensus groups have one distinctly tighter member, that "
    "member remains a bounded fallback but is not the strongest available evidence. Recovery "
    "is tried first because two disjoint multi-frame references can confirm a placement that "
    "the competing ordinary groups cannot settle. If recovery refuses, its evidence is "
    "recorded and the tighter ordinary group is still used; if recovery succeeds, its "
    "independently confirmed placement supersedes the fallback.",
    "The coarse ranking is a shape-level comparison on a decimated frame against a decimated "
    "reference. Its margin says the geometry distinguishes these placements at that "
    "resolution; a surface that repeats at a finer scale than the ranking can see is not "
    "detected here.",
    "TWO ambiguity verdicts are required and neither substitutes for the other. "
    "multiview.coarse_align's own verdict, at its own 1.15 margin on its own quantity, must "
    "call the ranking unambiguous; and the SELECTED hypothesis's coverage must clear the same "
    "1.15 margin over the best genuinely different placement still standing beside it. The "
    "second is asked as well because the two measure different things over different amounts "
    "of the frame: coarse_align ranks a DECIMATED cloud by a truncated mean nearest-neighbour "
    "distance on a grid 4% of the reference extent, and this module measures overlap over the "
    "WHOLE frame at the same tolerance every ordinary candidate's overlap is measured at. "
    "Measured on this repository's incompatible-minima fixture: 2.42 shared, 1.36 coverage, "
    "3.13 from the selected hypothesis. Both verdicts, every margin and coarse_align's "
    "limitations are recorded verbatim on every recovery including the accepted ones. A "
    "near-equal principal extent is recorded there as a warning about the PCA-derived seeds; "
    "it is not substituted for either ambiguity verdict, because it does not say that the "
    "shape's finer geometry or the explicit prior failed to distinguish the placements.",
    "The shared coarse verdict USED to be what refused this problem, and the correction went "
    "into the shared stage rather than around it. multiview.coarse_align scored a candidate on "
    "the mean nearest-neighbour distance over the points it MATCHED, so unmatched geometry "
    "contributed nothing and a placement that put most of the frame off the reference was "
    "scored on the fraction of itself it managed to place. Measured on this repository's "
    "incompatible-minima fixture, the right placement and a genuinely different one scored "
    "within 1.05x of each other while covering the reference 0.47 against 0.34, so the shared "
    "stage called a ranking ambiguous that the geometry distinguishes clearly, and this "
    "recovery refused a placeable frame. That score is now a TRUNCATED mean over ALL the "
    "sampled points - a point with no partner costs a bounded miss rather than nothing - and "
    "the same fixture reports 2.42. The change is in multiview, which relocalize, revisit, "
    "support and quality also read, and it moves their rankings too; it is not a "
    "tracker-private substitution for a shared verdict this module did not like.",
    "The selected hypothesis's margin is measured from the selected hypothesis. A "
    "higher-ranked hypothesis that a gate refused is still counted as a rival placement and "
    "still refuses the recovery; it never lends its own margin to a lower-ranked survivor. "
    "Only a hypothesis refinement carried into the SELECTED placement is excluded, because two "
    "seeds reaching one answer are one answer.",
    "The last accepted pose is offered to the coarse ranking as a prior. It is scored by the "
    "same decimated pass on the same grid as every generated orientation and is refused by "
    "every gate they are refused by; it is a starting pose, not evidence, and a recovery that "
    "selects it still has to be confirmed by a second disjoint reference that never saw it.",
    "A partition's viewpoint diversity is MEASURED, as the largest keyframe pose-gap between "
    "any two of its frames, and a partition under one keyframe motion gate is refused. Frames "
    "captured from one standstill are one view under several indices, and the frame-count gate "
    "alone cannot tell the difference.",
    "Both reference partitions are built from earlier ACCEPTED poses. They share no accepted "
    "frame with each other, which is what makes their agreement evidence about this placement "
    "rather than about one frame's own consistency - and it does not make either of them "
    "independent of an error the capture established before both of them.",
    "A recovery is a fallback for a frame the ordinary references could not agree on, not a "
    "relocalization. A frame whose true placement is further from the last accepted pose than "
    "the travel gate allows is refused here exactly as it is refused there.",
)

LIMITATIONS: Final = (
    "An accepted pose is agreement between bounded references under named thresholds, not a "
    "measurement of truth. Every reference here descends from earlier accepted poses, so a "
    "systematic error established early is agreed with rather than detected.",
    "Past the speed the capture guidance already calls too fast, the motion prior can carry a "
    "capture that would otherwise have refused. Measured on this repository's own scene, 16 "
    "frames at each step: at 30 degrees a frame it places all 16 within 2.32 degrees where the "
    "identity seed alone refuses at frame 7, and at 33 degrees it holds the sweep to 6.87 "
    "degrees where the identity seed alone places all 16 and is 41 degrees wrong from frame 7 "
    "on. But at 36 degrees a frame - roughly twice the surface travel at which travel_verdict "
    "already tells the operator to slow down - it placed all 16 with 23 degrees of error from "
    "frame 10, where the identity seed refused at frame 6 instead. The behaviour is not "
    "monotonic in speed (39 degrees returns to 2.13 degrees across all 16), because what "
    "changes is which local minimum one solve falls into. A seed cannot make a refusal safer; "
    "it can only decide where a solve starts, and past the guidance band a placement this "
    "module accepts is no more trustworthy than the geometry it was measured on.",
    "The motion prior is a constant-velocity extrapolation of ONE measured step, and the "
    "scanner is carried by hand rather than driven at a constant rate. It is a starting pose "
    "and is treated as one: it is not evidence, it is not recorded as a placement, it widens "
    "no gate, and a frame it seeds badly is refused exactly as an identity-seeded one is. It "
    "cannot describe an acceleration, it is not measured against an IMU because this unit has "
    "none, and it is deliberately absent - rather than approximated - wherever the chain it "
    "would be extrapolated from is not continuous.",
    "The model reference is a bounded sample of a bounded set of accepted frames, not the "
    "model. Overlap and residual against it are measured against that sample.",
    "Candidates are refinements from stated initial transforms. A placement wrong by more "
    "than refinement can travel is not searched for here; that is a relocalization.",
    "Keyframe selection is a threshold on a continuous measurement, so two runs whose poses "
    "differ in their fifth decimal - a different compute backend, for instance - can hold "
    "different keyframe sets, build different model references from them, and end with poses "
    "that differ by more than arithmetic would. Measured on the repository's own 24-frame "
    "fixture, python against scipy: the same origin and the same rescue decision through frame "
    "15, and the same keyframe set through frame 14, then a different retained keyframe at "
    "frame 15 - the first frame a lap-closing revisit can seed from - which propagates to "
    "every later frame's keyframe set. Clear of that threshold (the same fixture's first 15 "
    "frames) the two agree on every placement decision, with poses within 6.1e-4 of any "
    "rotation-matrix entry and 0.0505 depth units of translation.",
    "Convergence here means the solver's residual stopped moving by as much as a hundredth "
    "of the noise floor between iterations, not that the fit reached the global minimum. A "
    "fit that settles into a wrong local minimum is settled, and is refused - if it is "
    "refused at all - by the residual, overlap, travel and agreement evidence instead. Where "
    "that happens to every ordinary reference at once, a bounded coarse recovery may re-seed "
    "the refinement from a shape-level search; see RECOVERY_LIMITATIONS for what that does "
    "and does not establish.",
)


class TrackingNotAllowed(ScannerError):
    """A tracking request this module refuses to attempt."""

    status = DoctorStatus.UNSUPPORTED_MODE


@dataclass(frozen=True, slots=True)
class TrackingPolicy:
    """Every bound and gate one tracker runs under, as one recordable value.

    Schema-versioned and carried into the durable record because the live GUI
    and the offline reconstruction share this module: a reader comparing a
    capture's decisions with a reconstruction's has to be able to see that the
    two ran the same policy rather than assume it.

    ``use_keyframes`` and ``use_model`` exist for the ONE caller that has no
    accepted model to refer to - the tentative segment collected during a
    relocalization, which is chained against itself and nothing else. Turning
    them off does not loosen a gate; it removes references that do not exist.
    """

    max_keyframes: int = MAX_KEYFRAMES
    keyframe_candidates: int = KEYFRAME_CANDIDATES
    protected_recent_keyframes: int = PROTECTED_RECENT_KEYFRAMES
    max_reference_frames: int = MAX_REFERENCE_FRAMES
    model_reference_points: int = MODEL_REFERENCE_POINTS
    candidate_source_points: int = CANDIDATE_SOURCE_POINTS
    retained_cloud_points: int = RETAINED_CLOUD_POINTS
    max_retained_clouds: int = MAX_RETAINED_CLOUDS
    model_recent_frames: int = MODEL_RECENT_FRAMES
    min_track_points: int = MIN_TRACK_POINTS
    min_reference_overlap: float = MIN_REFERENCE_OVERLAP
    rescue_min_overlap: float = RESCUE_MIN_OVERLAP
    rescue_min_reference_frames: int = RESCUE_MIN_REFERENCE_FRAMES
    max_travel_multiple: float = MAX_TRAVEL_MULTIPLE
    convergence_noise_fraction: float = CONVERGENCE_NOISE_FRACTION
    agreement_cluster_margin: float = AGREEMENT_CLUSTER_MARGIN
    #: Whether an adjacent candidate that no accepted placement came out of may
    #: be retried once from the constant-velocity prediction. Turning it off
    #: removes a bounded second attempt and loosens nothing: the retry is judged
    #: by the same :meth:`IncrementalTracker._judge` the first attempt is, and
    #: it can only replace a candidate that was NOT accepted.
    motion_prior: bool = True
    max_prior_steps: int = MAX_PRIOR_STEPS
    use_keyframes: bool = True
    use_model: bool = True
    #: Whether an ordinary refusal may be offered to the bounded coarse
    #: recovery below. Off for the one caller with no accepted model to build
    #: disjoint partitions from - see :data:`ADJACENT_ONLY_POLICY` - and on
    #: everywhere else. Turning it off removes a fallback; it loosens nothing.
    recovery_enabled: bool = True
    recovery_source_points: int = RECOVERY_SOURCE_POINTS
    recovery_partitions: int = RECOVERY_PARTITIONS
    recovery_partition_frames: int = RECOVERY_PARTITION_FRAMES
    recovery_min_partition_frames: int = RECOVERY_MIN_PARTITION_FRAMES
    recovery_min_partition_pose_span: float = RECOVERY_MIN_PARTITION_POSE_SPAN
    recovery_partition_points: int = RECOVERY_PARTITION_POINTS
    recovery_ranked_hypotheses: int = RECOVERY_RANKED_HYPOTHESES
    recovery_hypotheses: int = RECOVERY_HYPOTHESES
    recovery_ambiguity_margin: float = RECOVERY_AMBIGUITY_MARGIN
    recovery_coarse_distance_multiple: float = RECOVERY_COARSE_DISTANCE_MULTIPLE

    def __post_init__(self) -> None:
        """Refuse a policy whose own bounds do not hold, before one is ever used.

        Every check below is a RELATIONSHIP rather than a range, and each is
        here because the round-1 policy admitted a combination that broke a
        promise this module makes elsewhere: a keyframe set that could not be
        brought back inside its bound because the anchor plus the protected tail
        already exceeded it, a retention budget too small to hold the keyframes
        and the current reference the selection rules assume are there, and a
        reference point budget smaller than its own frame count, which reported
        frames whose every point had been truncated back off the reference.

        Refusing here rather than at the point of use is the whole value: a
        tracker that exists at all is a tracker whose hard bounds are reachable.
        """
        if self.min_track_points < MIN_TRACK_POINTS:
            raise TrackingNotAllowed(
                f"a placement needs at least {MIN_TRACK_POINTS} points - the fewest that can "
                f"span a plane - got min_track_points {self.min_track_points}"
            )
        if self.max_keyframes < 1:
            raise TrackingNotAllowed(
                f"a tracker needs at least one keyframe slot for the anchor, got "
                f"{self.max_keyframes}"
            )
        if self.protected_recent_keyframes < 0:
            raise TrackingNotAllowed(
                f"protected_recent_keyframes cannot be negative, got "
                f"{self.protected_recent_keyframes}"
            )
        if self.max_keyframes <= self.protected_recent_keyframes:
            # The anchor is never evicted and neither are the newest
            # ``protected_recent_keyframes``. If those alone reach the bound
            # there is no evictable entry, and the set stays permanently over a
            # bound this class advertises as hard.
            raise TrackingNotAllowed(
                f"max_keyframes {self.max_keyframes} is not more than the "
                f"{self.protected_recent_keyframes} protected recent keyframes plus the anchor, "
                "so the set could never be brought back inside its own bound"
            )
        if self.max_reference_frames < 1:
            raise TrackingNotAllowed(
                f"a model reference needs at least one frame, got {self.max_reference_frames}"
            )
        if self.model_reference_points < 1 or self.candidate_source_points < 1:
            raise TrackingNotAllowed(
                "a reference and a candidate source both need a positive point budget, got "
                f"{self.model_reference_points} and {self.candidate_source_points}"
            )
        if self.model_reference_points < self.max_reference_frames:
            # ``_model_reference`` spends ``model_reference_points // frames``
            # on each selected frame. Below one point per frame the reference
            # would name frames whose contribution was truncated away, and
            # ``rescue_min_reference_frames`` is counted against that name.
            raise TrackingNotAllowed(
                f"model_reference_points {self.model_reference_points} is under the "
                f"{self.max_reference_frames} frames a reference may be built from, so a frame "
                "could be named in a reference it contributed no point to"
            )
        if self.retained_cloud_points < self.min_track_points:
            raise TrackingNotAllowed(
                f"retained_cloud_points {self.retained_cloud_points} is under the "
                f"{self.min_track_points} points a placement can be built on, so every accepted "
                "cloud would be thinned below usefulness as it was retained"
            )
        if self.candidate_source_points < self.min_track_points:
            raise TrackingNotAllowed(
                f"candidate_source_points {self.candidate_source_points} is under the "
                f"{self.min_track_points} points a placement can be built on"
            )
        if self.max_retained_clouds <= self.max_keyframes:
            # Retention has to hold every keyframe AND the frame the next
            # adjacent candidate is fitted against. A smaller budget silently
            # released the anchor or the current reference, which the keyframe
            # and model selection rules both assume are present.
            raise TrackingNotAllowed(
                f"max_retained_clouds {self.max_retained_clouds} is not more than the "
                f"{self.max_keyframes} keyframes it has to hold alongside the previous accepted "
                "frame, so a cloud the policy promises to retain would be released"
            )
        if self.keyframe_candidates < 0 or self.model_recent_frames < 0:
            raise TrackingNotAllowed("candidate and recency counts cannot be negative")
        if self.max_prior_steps < 1:
            # One step is the ordinary next frame. Below that the prediction
            # could never be formed for any frame at all, which is a policy
            # asking for a feature it has switched off by arithmetic rather than
            # by ``motion_prior``.
            raise TrackingNotAllowed(
                f"max_prior_steps {self.max_prior_steps} is under the one delivery step an "
                "ordinary next frame is, so the prediction could never be formed at all; set "
                "motion_prior False to switch it off"
            )
        for value, name in (
            (self.min_reference_overlap, "min_reference_overlap"),
            (self.rescue_min_overlap, "rescue_min_overlap"),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise TrackingNotAllowed(f"{name} {value!r} is not a fraction in (0, 1]")
        if self.rescue_min_overlap < self.min_reference_overlap:
            raise TrackingNotAllowed(
                f"rescue_min_overlap {self.rescue_min_overlap} is below the ordinary "
                f"{self.min_reference_overlap} overlap gate, which would make the placement with "
                "no second opinion the easiest one to accept"
            )
        if self.rescue_min_reference_frames < MIN_MODEL_REFERENCE_FRAMES:
            raise TrackingNotAllowed(
                f"rescue_min_reference_frames {self.rescue_min_reference_frames} is under the "
                f"{MIN_MODEL_REFERENCE_FRAMES} viewpoints a model reference must span before it "
                "is a model reference at all"
            )
        if not math.isfinite(self.max_travel_multiple) or self.max_travel_multiple <= 0.0:
            raise TrackingNotAllowed(
                f"max_travel_multiple {self.max_travel_multiple!r} must be a finite positive "
                "multiple of the noise floor"
            )
        if (
            not math.isfinite(self.convergence_noise_fraction)
            or self.convergence_noise_fraction <= 0.0
        ):
            raise TrackingNotAllowed(
                f"convergence_noise_fraction {self.convergence_noise_fraction!r} must be a "
                "finite positive fraction of the noise floor"
            )
        if not math.isfinite(self.agreement_cluster_margin) or self.agreement_cluster_margin <= 1.0:
            raise TrackingNotAllowed(
                f"agreement_cluster_margin {self.agreement_cluster_margin!r} must be a finite "
                "margin above 1.0; at or below it a tie between equally tight consensus "
                "clusters would be called a clear winner"
            )
        self._check_recovery()

    def _check_recovery(self) -> None:
        """Refuse a recovery budget that could not keep its own promises.

        Separate from the rest only because ``__post_init__`` was already the
        length of a small function; every check here is the same kind of
        RELATIONSHIP the ones above are, and each names the promise it protects.
        A policy with ``recovery_enabled`` false is checked too: a bound nobody
        currently spends is still a bound a caller can read, and one that only
        becomes false when it is switched on is a trap rather than a bound.
        """
        if self.recovery_partitions < RECOVERY_PARTITIONS:
            # One partition cannot confirm itself. The whole claim a recovery
            # makes is that two references sharing no accepted frame put the
            # frame in the same place, and with one reference there is no
            # second placement for the first to agree with.
            raise TrackingNotAllowed(
                f"recovery_partitions {self.recovery_partitions} is under the "
                f"{RECOVERY_PARTITIONS} disjoint references a recovery has to agree between; "
                "a placement confirmed only by the reference that produced it is not confirmed"
            )
        if self.recovery_min_partition_frames < MIN_MODEL_REFERENCE_FRAMES:
            raise TrackingNotAllowed(
                f"recovery_min_partition_frames {self.recovery_min_partition_frames} is under "
                f"the {MIN_MODEL_REFERENCE_FRAMES} accepted viewpoints a reference must span "
                "before it is more than one view wearing a broader name"
            )
        if self.recovery_partition_frames < self.recovery_min_partition_frames:
            raise TrackingNotAllowed(
                f"recovery_partition_frames {self.recovery_partition_frames} is under the "
                f"{self.recovery_min_partition_frames} frames every partition must span, so no "
                "partition this policy could build would ever be allowed to confirm anything"
            )
        if self.recovery_partition_points < self.recovery_partition_frames:
            # Same arithmetic as ``model_reference_points``: a partition spends
            # ``points // frames`` on each frame, and below one point per frame
            # it would name a viewpoint it carries no point from.
            raise TrackingNotAllowed(
                f"recovery_partition_points {self.recovery_partition_points} is under the "
                f"{self.recovery_partition_frames} frames a partition may be built from, so a "
                "frame could be named in a partition it contributed no point to"
            )
        if (
            not math.isfinite(self.recovery_min_partition_pose_span)
            or self.recovery_min_partition_pose_span <= 0.0
        ):
            raise TrackingNotAllowed(
                "recovery_min_partition_pose_span "
                f"{self.recovery_min_partition_pose_span!r} must be a finite positive number of "
                "keyframe motion gates: a partition whose frames are allowed to span no "
                "viewpoint change at all is a partition of one view under several indices."
            )
        if self.recovery_source_points < self.min_track_points:
            raise TrackingNotAllowed(
                f"recovery_source_points {self.recovery_source_points} is under the "
                f"{self.min_track_points} points a placement can be built on"
            )
        if self.recovery_hypotheses < 1 or self.recovery_ranked_hypotheses < 1:
            raise TrackingNotAllowed(
                "a recovery needs at least one ranked and one refined hypothesis, got "
                f"{self.recovery_ranked_hypotheses} and {self.recovery_hypotheses}"
            )
        if self.recovery_hypotheses > self.recovery_ranked_hypotheses:
            raise TrackingNotAllowed(
                f"recovery_hypotheses {self.recovery_hypotheses} is over the "
                f"{self.recovery_ranked_hypotheses} the ranking is allowed to hold, so the "
                "refinement budget could not be spent"
            )
        if (
            not math.isfinite(self.recovery_ambiguity_margin)
            or self.recovery_ambiguity_margin <= 1.0
        ):
            raise TrackingNotAllowed(
                f"recovery_ambiguity_margin {self.recovery_ambiguity_margin!r} must be a finite "
                "margin above 1.0; at or below it every tie counts as a clear winner"
            )
        if (
            not math.isfinite(self.recovery_coarse_distance_multiple)
            or self.recovery_coarse_distance_multiple < 1.0
        ):
            raise TrackingNotAllowed(
                f"recovery_coarse_distance_multiple {self.recovery_coarse_distance_multiple!r} "
                "must be a finite multiple of at least 1.0 of the tracker's own correspondence "
                "distance; below it the wide pass would be narrower than the judged one"
            )

    @property
    def max_candidates(self) -> int:
        """The ceiling on ICP solves the ORDINARY decision for one frame can cost.

        One more than :attr:`ordinary_candidates` while ``motion_prior`` is on,
        and the difference is a CEILING rather than a cost. The adjacent retry
        runs only where the identity-seeded attempt produced no accepted
        candidate, so a frame every reference places at the first attempt still
        pays exactly the references it is judged by - which is what a long
        capture pays on almost every frame.
        """
        return self.ordinary_candidates + int(self.motion_prior)

    @property
    def ordinary_candidates(self) -> int:
        """Solves a frame costs when the adjacent reference places it first time."""
        return 1 + (self.keyframe_candidates if self.use_keyframes else 0) + int(self.use_model)

    @property
    def max_recovery_reference_frames(self) -> int:
        """Accepted frames one recovery may inspect, however long the capture ran."""
        if not self.recovery_enabled:
            return 0
        return self.recovery_partitions * self.recovery_partition_frames

    @property
    def max_recovery_orientations(self) -> int:
        """Orientations one recovery's single coarse search may rank.

        The search's own structural ceiling, not this policy's shortlist: see
        :data:`COARSE_RANKED_POSES` and :attr:`recovery_ranked_hypotheses`.
        """
        if not self.recovery_enabled:
            return 0
        return COARSE_RANKED_POSES

    @property
    def max_recovery_solves(self) -> int:
        """The ceiling on ICP solves one recovery ATTEMPT can cost.

        Two passes per refined hypothesis - the wide seeding pass and the judged
        one - plus ONE per confirming partition, which is seeded from a
        placement that already refined and so needs no wide pass of its own.
        Additional to :attr:`max_candidates`, never instead of it: a recovery is
        reached only after the ordinary decision has already been made and
        either refused or retained a distinctly tighter tied cluster as its
        bounded fallback.
        """
        if not self.recovery_enabled:
            return 0
        return 2 * self.recovery_hypotheses + (self.recovery_partitions - 1)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "noise_floor": NOISE_FLOOR,
            "max_keyframes": self.max_keyframes,
            "keyframe_candidates": self.keyframe_candidates,
            "protected_recent_keyframes": self.protected_recent_keyframes,
            "max_reference_frames": self.max_reference_frames,
            "model_reference_points": self.model_reference_points,
            "candidate_source_points": self.candidate_source_points,
            "retained_cloud_points": self.retained_cloud_points,
            "max_retained_clouds": self.max_retained_clouds,
            "model_recent_frames": self.model_recent_frames,
            "max_candidates": self.max_candidates,
            "ordinary_candidates": self.ordinary_candidates,
            "motion_prior": self.motion_prior,
            "max_prior_steps": self.max_prior_steps,
            "min_track_points": self.min_track_points,
            "min_reference_overlap": self.min_reference_overlap,
            "rescue_min_overlap": self.rescue_min_overlap,
            "rescue_min_reference_frames": self.rescue_min_reference_frames,
            "max_travel_multiple": self.max_travel_multiple,
            "max_travel": self.max_travel_multiple * NOISE_FLOOR,
            "convergence_noise_fraction": self.convergence_noise_fraction,
            "convergence_threshold": self.convergence_noise_fraction * NOISE_FLOOR,
            "keyframe_min_degrees": KEYFRAME_MIN_DEGREES,
            "keyframe_min_distance": KEYFRAME_MIN_DISTANCE_MULTIPLE * NOISE_FLOOR,
            "keyframe_novel_overlap": KEYFRAME_NOVEL_OVERLAP,
            "stationary_degrees": STATIONARY_DEGREES,
            "stationary_distance": STATIONARY_DISTANCE_MULTIPLE * NOISE_FLOOR,
            "agreement_residual_multiple": AGREEMENT_RESIDUAL_MULTIPLE,
            "max_disagreement": AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR,
            "agreement_cluster_margin": self.agreement_cluster_margin,
            "overlap_tolerance": OVERLAP_TOLERANCE_MULTIPLE * NOISE_FLOOR,
            "use_keyframes": self.use_keyframes,
            "use_model": self.use_model,
            "recovery": {
                "enabled": self.recovery_enabled,
                "source_points": self.recovery_source_points,
                "partitions": self.recovery_partitions,
                "partition_frames": self.recovery_partition_frames,
                "min_partition_frames": self.recovery_min_partition_frames,
                "min_partition_pose_span": self.recovery_min_partition_pose_span,
                "partition_points": self.recovery_partition_points,
                "max_reference_frames": self.max_recovery_reference_frames,
                "max_reference_points": (self.recovery_partitions * self.recovery_partition_points),
                "priors_offered": RECOVERY_PRIORS if self.recovery_enabled else 0,
                "max_coarse_poses_ranked": (COARSE_RANKED_POSES if self.recovery_enabled else 0),
                "max_coarse_hypotheses_scored": self.recovery_ranked_hypotheses,
                "refined_hypotheses": self.recovery_hypotheses,
                "max_confirmations_refined": self.recovery_partitions - 1,
                "max_icp_solves": self.max_recovery_solves,
                "max_coarse_searches": 1 if self.recovery_enabled else 0,
                "ambiguity_margin": self.recovery_ambiguity_margin,
                "coarse_distance_multiple": self.recovery_coarse_distance_multiple,
            },
            "local_chain": local_chain_gates(),
        }


#: The default every ordinary capture and every offline reconstruction uses.
DEFAULT_POLICY: Final = TrackingPolicy()

#: What the tentative segment of a relocalization runs under: the same gates,
#: with the references it does not have switched off.
ADJACENT_ONLY_POLICY: Final = TrackingPolicy(
    use_keyframes=False, use_model=False, recovery_enabled=False
)


def _json_number(value: float) -> float | None:
    """``value``, or ``None`` where JSON has no word for it.

    The same convention :func:`scanner3d.relocalize._json_number` uses and for
    the same reason: these numbers reach a durable record written with
    ``allow_nan=False``, and "not measurable" is the honest word rather than a
    substituted finite one.
    """
    return value if math.isfinite(value) else None


def _disagreements(
    survivors: Sequence[TrackCandidate], points: Sequence[Point], gate: float
) -> list[tuple[TrackCandidate, TrackCandidate, float]]:
    """Every PAIR of surviving candidates that placed ``points`` further than ``gate`` apart.

    Every pair, and that is the whole point of this being its own function.
    Round 1 compared each survivor against the PREFERRED one only, which accepts
    a set whose two extremes are nearly twice the gate apart as long as each sits
    near the third - so "the references agreed" could be true of a set that
    contained two answers. Agreement has to mean the survivors form one cluster,
    and pairwise is the only reading of that which does not depend on which
    candidate happened to be preferred.

    Returned rather than reduced to a boolean so the refusal can name which
    references disagreed and by how much; an operator can act on "the model and
    the previous frame are 9 units apart" and cannot act on "refused".
    """
    return [
        (a, b, apart)
        for a, b in itertools.combinations(survivors, 2)
        if (apart := _disagreement(a.pose, b.pose, points)) > gate
    ]


def _agreement_clusters(
    survivors: Sequence[TrackCandidate], points: Sequence[Point], gate: float
) -> list[tuple[TrackCandidate, ...]]:
    """Every LARGEST set of survivors that all agree with EACH OTHER, not merely with a leader.

    Mutual agreement, and that word is the whole of it. Round 1 measured each
    survivor against the preferred candidate alone, so a set whose two extremes
    were nearly twice the gate apart passed as long as each sat near the third.
    Here a set counts only if every pair inside it is within ``gate``, which is
    the property "these references agree" was always supposed to name.

    Returns every maximum-size such set, so the caller can distinguish one
    agreement from tied groups. The caller may select a tied group only when its
    own widest pose disagreement is distinctly tighter than the runner-up's;
    residual never breaks the tie.

    Brute force over subsets, and that is not a shortcut: the number of
    survivors is bounded by :attr:`TrackingPolicy.max_candidates`, which is
    three for the shared policy, so this enumerates at most seven sets.
    """
    best: list[tuple[TrackCandidate, ...]] = []
    for size in range(len(survivors), 0, -1):
        for subset in itertools.combinations(survivors, size):
            if all(
                _disagreement(a.pose, b.pose, points) <= gate
                for a, b in itertools.combinations(subset, 2)
            ):
                best.append(subset)
        if best:
            return best
    return best


def _disagreement(a: Rigid, b: Rigid, points: Sequence[Point]) -> float:
    """How far apart two placements of the same frame put its surface.

    :func:`scanner3d.relocalize._same_placement` answers the same question with
    an angle and a distance compared separately, and that is right for what it
    decides: two coarse seeds either fell into one minimum or they did not, and
    the segment being placed has no single scale. Here the frame IS the scale -
    these are its own points - so the honest measurement is what the two
    placements do to them, which is one number in depth units and directly
    comparable with the residual gate they both had to clear.
    """
    return frame_travel(points, a.inverse().compose(b))


def convergence_tolerance(target: Sequence[Point], *, fraction: float) -> float:
    """The solver ``tolerance`` that means "settled to ``fraction`` of the noise floor".

    :func:`~scanner3d.icp_plane.icp_point_to_plane` and
    :func:`~scanner3d.registration.icp` both express ``tolerance`` as a fraction
    of the TARGET cloud's bounding-box diagonal, because this repository has no
    physical unit for a length. This decision does have one: a residual is a
    distance in depth units and :data:`~scanner3d.pipeline.NOISE_FLOOR` is the
    smallest such distance that means anything on this sensor. So the threshold
    is chosen absolutely - ``fraction * NOISE_FLOOR`` - and divided back through
    the target's own extent here, which is the same quantity the solver will
    multiply it by. The result is one absolute convergence threshold whatever
    the object's size, deterministic, and recorded in
    :meth:`TrackingPolicy.as_dict` as ``convergence_threshold``.

    A target with no extent - every point identical, which
    :func:`~scanner3d.icp_plane.icp_point_to_plane` refuses seconds later anyway
    - has no fraction to express, so the solver's own default is returned rather
    than a division by zero or an infinite tolerance that would call the first
    iteration converged.
    """
    extent = cloud_extent(target)
    if not math.isfinite(extent) or extent <= 0.0:
        return DEFAULT_TOLERANCE
    return (fraction * NOISE_FLOOR) / extent


def bounded_source(
    cloud: Sequence[Point], policy: TrackingPolicy = DEFAULT_POLICY
) -> tuple[Point, ...]:
    """``cloud`` thinned to the points any candidate is allowed to be fitted FROM.

    Public because the offline reconstruction pre-fits every adjacent pair in
    worker processes before a tracker exists to hand them to, and a pre-fit of
    a differently sampled cloud is a fit of a different problem. One function,
    so the live path and the offline path cannot drift apart.
    """
    return _sampled(tuple(cloud), policy.candidate_source_points)


def retained_cloud(
    cloud: Sequence[Point], policy: TrackingPolicy = DEFAULT_POLICY
) -> tuple[Point, ...]:
    """``cloud`` thinned to what an accepted frame is allowed to keep.

    Public for the same reason as :func:`bounded_source`: the previous accepted
    cloud is the adjacent candidate's reference, and the offline path has to
    pre-fit against exactly the cloud the tracker will have retained.
    """
    return _sampled(tuple(cloud), policy.retained_cloud_points)


def _usable_point(point: object) -> bool:
    """Whether one entry of an offered cloud is a :data:`~scanner3d.geometry.Point`.

    Exactly three coordinates, each one a real finite number. The type says
    ``tuple[float, float, float]`` and this asks whether the object actually IS
    that, because a caller outside the type checker - a replay reading a file, a
    Qt callback, a test double - can hand over anything at all.

    ``bool`` is refused even though it is an ``int`` subclass. The contract is
    three coordinates; ``True`` is a flag that would silently become the number
    1, and a cloud carrying flags where coordinates should be is a mistake to
    report rather than to arithmetically absorb. ``complex`` is refused for the
    same reason: ``math.isfinite`` accepts neither, but ``isinstance`` has to
    say so first or the check raises instead of answering.
    """
    if isinstance(point, str | bytes | bytearray) or not isinstance(point, tuple | list):
        return False
    if len(point) != 3:
        return False
    for value in point:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        if not math.isfinite(value):
            return False
    return True


def _finite_cloud(points: Sequence[Point]) -> bool:
    """Whether ``points`` is a usable cloud: every entry three finite real coordinates.

    A TOTAL validator, and that is the whole of the change from round 1. Round 1
    asked ``math.isfinite`` of whatever iterating a point produced, which answers
    the question only for input that was already well formed: a two- or
    four-coordinate entry passed, and a string or ``None`` coordinate RAISED out
    of :meth:`IncrementalTracker.offer` - the one method whose contract is that
    it never raises for ordinary data - instead of refusing. Here every
    malformed shape returns ``False``, so the caller refuses the frame with its
    evidence and the tracker mutates nothing.

    Checked BEFORE anything is fitted or admitted, including for a seeded frame
    that is not fitted at all: a bad coordinate reaching the accepted model
    poisons every later reference built from it, every overlap measured against
    it and every eviction distance computed from it, and the frame that finally
    reports a NaN is not the frame that carried one in.
    """
    return all(_usable_point(point) for point in points)


def _rigid_finite(transform: Rigid) -> bool:
    """Whether a pose is made of real numbers, rotation as well as translation.

    :mod:`scanner3d.relocalize` asks the same question of a fit's transform and
    keeps its own private answer; this is the same check applied to a pose a
    caller SUPPLIED rather than one a solver produced, which is the only kind
    that reaches the accepted model without passing an ICP gate first.
    """
    return all(math.isfinite(value) for row in transform.rotation for value in row) and all(
        math.isfinite(value) for value in transform.translation
    )


def _sampled(cloud: Sequence[Point], limit: int, *, offset: int = 0) -> tuple[Point, ...]:
    """``cloud`` thinned to at most ``limit`` points, evenly and deterministically.

    ``offset`` shifts which points a stride keeps, and it is the difference
    between a model reference that is a surface and one that is a stencil.
    Several accepted frames of the same object, each thinned by the same stride
    from the same starting index, contribute the SAME points - so a reference
    assembled from eight of them held only one frame's worth of distinct
    geometry, and a frame fitted against it overlapped about 0.6 of itself
    however dense the budget said the reference was. Measured before and after:
    on a 460-point object sampled by eight frames, overlap went from 0.62 to
    effectively complete. Staggering the start is deterministic, costs nothing,
    and is why the model candidate can be as precise as the adjacent one.
    """
    if limit <= 0 or not cloud:
        return ()
    if len(cloud) <= limit:
        return tuple(cloud)
    # CEILING division, and this is the second half of the same bug. With a
    # floor, a 460-point cloud asked for 300 gets a stride of 1 and is then
    # truncated to the first 300 - which for any cloud whose points are in
    # generation order rather than shuffled is a CONTIGUOUS REGION of the
    # object, not a thinning of it. Measured on this repository's own lobed
    # ellipsoid fixture, that made a model reference cover about six tenths of
    # the surface and report exactly that as its overlap. A ceiling spends a
    # little less of the budget and covers the whole cloud.
    step = max(1, -(-len(cloud) // limit))
    return tuple(cloud)[offset % step :: step][:limit]


@dataclass(frozen=True, slots=True)
class TrackCandidate:
    """One independent placement of one frame, and every number it was judged on.

    ``pose`` is always a WORLD pose - the transform carrying this frame's own
    coordinates into the accepted model's frame - whatever reference produced
    it. The composition that gets there differs per origin and is done once, in
    :meth:`IncrementalTracker._candidate`, so no reader or caller ever has to
    know which direction a particular ICP solve reported.
    """

    origin: str
    #: The accepted frame indices this candidate was fitted against. One entry
    #: for an adjacent or keyframe candidate; several for a model candidate,
    #: and the length of it is what
    #: :data:`RESCUE_MIN_REFERENCE_FRAMES` is measured against.
    reference_frames: tuple[int, ...]
    reference_points: int
    source_points: int
    pose: Rigid
    #: The four shared alignment measurements, judged by the same function the
    #: tentative local chain uses.
    edge: LocalEdge
    overlap: float
    travel: float
    #: True exactly when :attr:`edge` cleared every shared alignment gate AND
    #: the overlap and travel gates this decision adds. There is deliberately no
    #: third state: round 1 carried "caveats" - gates a candidate failed but was
    #: accepted in spite of - and that made ``accepted`` disagree with
    #: ``edge.accepted`` and let a failed fit be counted as agreement.
    accepted: bool
    refusals: Refusals
    #: Which of the two gates THIS decision adds - :data:`OVERLAP_GATE`,
    #: :data:`TRAVEL_GATE` - this candidate failed, named rather than left to be
    #: parsed back out of :attr:`refusals`. The shared alignment gates are
    #: already named one by one in :attr:`edge`; these two were not named
    #: anywhere, and the recovery trigger has to be able to tell a placement
    #: that is merely UNRESOLVED from one every reference agrees is a jump.
    failed_gates: tuple[str, ...] = ()
    #: What this candidate's solve was started from - one of
    #: :data:`IDENTITY_SEED`, :data:`PRIOR_SEED`, :data:`PREDICTED_SEED`,
    #: :data:`COARSE_SEED` or :data:`RECOVERED_SEED`. Evidence, not a gate:
    #: nothing reads it back to decide anything, and an accepted candidate is
    #: accepted on the same measurements whichever seed reached it.
    seed: str = IDENTITY_SEED
    #: How far this placement sits from where the last accepted motion
    #: predicted the frame, in the same units as ``travel``; None where no
    #: prediction existed. The travel gate is judged against this where it is
    #: present, so a steady fast step reads large in ``travel`` and small here.
    travel_from_prediction: float | None = None

    @property
    def rmse(self) -> float:
        return self.edge.rmse

    def as_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "reference_frames": list(self.reference_frames),
            "reference_points": self.reference_points,
            "source_points": self.source_points,
            "pose": self.pose.as_dict(),
            "alignment": self.edge.as_dict(),
            "overlap": _json_number(self.overlap),
            "travel": _json_number(self.travel),
            "travel_from_prediction": (
                None
                if self.travel_from_prediction is None
                else _json_number(self.travel_from_prediction)
            ),
            "accepted": self.accepted,
            "refusals": list(self.refusals),
            "failed_gates": list(self.failed_gates),
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class RecoveryPartition:
    """One bounded set of accepted frames a coarse recovery is measured against.

    Disjoint from every other partition of the same recovery by construction -
    see :meth:`IncrementalTracker._recovery_partitions` - which is the whole of
    what makes the second one a CONFIRMATION rather than a second reading of the
    first. Recorded with its frames rather than only its size, because "two
    references agreed" is only a claim about independence if a reader can see
    that they shared no accepted frame.
    """

    frames: tuple[int, ...]
    points: int
    #: The largest :meth:`IncrementalTracker._pose_distance` between any two of
    #: :attr:`frames`, in keyframe motion gates - the MEASURED viewpoint
    #: diversity this partition actually spans, against
    #: :data:`RECOVERY_MIN_PARTITION_POSE_SPAN`. Recorded rather than implied by
    #: the frame count, because four indices captured from one standstill are
    #: one viewpoint and this is the only number that says so.
    pose_span: float

    def as_dict(self) -> dict[str, object]:
        return {
            "frames": list(self.frames),
            "points": self.points,
            "pose_span": _json_number(self.pose_span),
        }


@dataclass(frozen=True, slots=True)
class RecoveryHypothesis:
    """One coarse initial pose, its shape-level rank, and what refinement made of it.

    Kept for every hypothesis the recovery ranked in its refinement budget,
    accepted or not. A recovery that refused because three of its four
    hypotheses were jumps and the fourth never converged is a different event
    from one that refused because two survived and disagreed, and a record that
    reduced both to "recovery refused" could not tell them apart.
    """

    rank: int
    coarse_origin: str
    #: Coverage of the offered frame against the searching partition at this
    #: hypothesis's COARSE pose, before any refinement. The quantity the
    #: ambiguity margin is measured on - see
    #: :data:`RECOVERY_AMBIGUITY_MARGIN`.
    coarse_overlap: float
    #: :attr:`~scanner3d.multiview.AlignmentCandidate.score` as the coarse stage
    #: reported it, carried unchanged so a reader can see the saturation the
    #: ranking had to work around rather than take this module's word for it.
    coarse_score: float
    initial: Rigid
    refined: Rigid | None
    accepted: bool
    refusals: Refusals

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "coarse_origin": self.coarse_origin,
            "coarse_overlap": _json_number(self.coarse_overlap),
            "coarse_score": _json_number(self.coarse_score),
            "initial": self.initial.as_dict(),
            "refined": self.refined.as_dict() if self.refined is not None else None,
            "accepted": self.accepted,
            "refusals": list(self.refusals),
        }


@dataclass(frozen=True, slots=True)
class RecoveryWork:
    """What one recovery ATTEMPT actually cost, counted rather than promised.

    Every field here has a ceiling in :meth:`TrackingPolicy.as_dict` under
    ``recovery``, and the two are recorded side by side on purpose: a bound is
    only a bound if the record carries the number to check it against.

    The coarse stage and the refinement stage are counted SEPARATELY throughout,
    because they do not run on the same data and a single number for either
    would be a smaller number than the truth. The ranking is scored from
    :attr:`coarse_source_points` of the frame against one partition; every
    refinement is fitted from :attr:`refine_source_points` of it - the same
    sample every ordinary candidate was fitted from - against a partition each.
    Likewise the coarse search RANKS :attr:`coarse_orientations_ranked`
    orientations however short the shortlist it is asked for is, and only
    :attr:`coarse_hypotheses_scored` of those are then scored for coverage.
    """

    #: Points of the offered frame the coarse RANKING was scored from.
    coarse_source_points: int
    #: Points of the offered frame every REFINEMENT and confirmation was fitted
    #: from. Not the same sample as above and usually larger, which is why
    #: reporting one of the two as "the recovery's source points" would be a
    #: false statement about the work whichever of them was chosen.
    refine_source_points: int
    coarse_searches: int
    #: Orientations the coarse search itself ranked, bounded by
    #: :data:`COARSE_RANKED_POSES` rather than by the shortlist below.
    coarse_orientations_ranked: int
    #: Of those, the shortlist this module measured coverage for, bounded by
    #: :attr:`TrackingPolicy.recovery_ranked_hypotheses`.
    coarse_hypotheses_scored: int
    hypotheses_refined: int
    #: Refinements of the surviving placement against the CONFIRMING partitions,
    #: counted apart from the hypotheses because they are a different question
    #: asked of different references.
    confirmations_refined: int
    icp_solves: int
    partitions: int
    reference_frames: int
    reference_points: int
    #: Points of the ONE partition the coarse search and every hypothesis
    #: refinement ran against, apart from the total above.
    coarse_reference_points: int

    def as_dict(self) -> dict[str, object]:
        return {
            "coarse_source_points": self.coarse_source_points,
            "refine_source_points": self.refine_source_points,
            "coarse_searches": self.coarse_searches,
            "coarse_orientations_ranked": self.coarse_orientations_ranked,
            "coarse_hypotheses_scored": self.coarse_hypotheses_scored,
            "hypotheses_refined": self.hypotheses_refined,
            "confirmations_refined": self.confirmations_refined,
            "icp_solves": self.icp_solves,
            "partitions": self.partitions,
            "reference_frames": self.reference_frames,
            "reference_points": self.reference_points,
            "coarse_reference_points": self.coarse_reference_points,
        }


@dataclass(frozen=True, slots=True)
class RecoveryEvidence:
    """Why a bounded coarse recovery was reached, what it ranked, and what it decided.

    Schema-versioned on its own, apart from :data:`SCHEMA_VERSION`, because this
    is the record a reader compares between a live capture and the offline
    reconstruction of the same frames: the two run the same code, and this is
    how a reader can see that rather than assume it.

    ``attempted`` false means the ordinary refusal was NOT offered to the
    recovery at all, and :attr:`refusals` says which rule kept it out. That is
    deliberately not the same value as an attempt that ran and refused: "the
    evidence already proved this frame is a jump" and "four hypotheses were
    ranked and none of them held up" are different statements about the capture.
    """

    schema_version: int
    triggered_by: str | None
    attempted: bool
    accepted: bool
    partitions: tuple[RecoveryPartition, ...]
    #: The best coarse coverage, the best GENUINELY DIFFERENT one, and the ratio
    #: between them. Preserved whatever refinement went on to do, because
    #: refinement can merge two seeds into one pose and it cannot make a shape
    #: that sits several ways sit one way.
    best_coarse_overlap: float
    rival_coarse_overlap: float | None
    coarse_overlap_margin: float
    coarse_overlap_unambiguous: bool
    #: :func:`~scanner3d.multiview.coarse_align`'s OWN ambiguity verdict and
    #: margin, carried verbatim - and REQUIRED, not replaced.
    #:
    #: Round 1 recorded this verdict and then permitted an acceptance while it
    #: was false, on the argument that the shared quantity - then a mean
    #: nearest-neighbour distance over MATCHED points only - could not tell the
    #: placements apart. That was a substitution of one module's private
    #: vocabulary for the shared one, and it is withdrawn. A recovery needs
    #: BOTH: this verdict true, AND the coverage margin below.
    #:
    #: The complaint behind the substitution was real, and it was fixed where it
    #: belonged. That score ignored unmatched geometry entirely, so a placement
    #: holding a tenth of the frame tidily on the reference outscored one
    #: holding the whole frame roughly on it. It is now a truncated mean over
    #: ALL the sampled points - see
    #: :data:`~scanner3d.multiview.SCORE_MISS_CELLS` - so coverage is part of
    #: the shared ordering, and the incompatible-minima fixture that reported a
    #: 1.05 margin reports 2.42.
    #:
    #: Coverage is still asked as well, because it is a different measurement
    #: over a different amount of the frame: this margin comes from a decimated
    #: cloud on a coarse grid, and the coverage margin comes from the whole
    #: frame at the same tolerance the accepted placement's own overlap will be
    #: judged at. A reader comparing the two numbers on any recovery - accepted
    #: or refused - can see both.
    coarse_score_margin: float
    coarse_score_unambiguous: bool
    #: :attr:`~scanner3d.multiview.CoarseAlignment.limitations` verbatim,
    #: including that function's own sentence when it judged the ranking
    #: ambiguous.
    coarse_limitations: tuple[str, ...]
    coarse_source_degenerate: bool
    coarse_reference_degenerate: bool
    hypotheses: tuple[RecoveryHypothesis, ...]
    #: The ambiguity evidence for the hypothesis that was ACTUALLY SELECTED,
    #: which is not always the coarse ranking's top entry and may not borrow its
    #: margin when it is not.
    #:
    #: Round 1 carried the best hypothesis's margin as the authority for
    #: whichever hypothesis survived refinement. Measured on
    #: ``20260828-105933`` at ``frame_0032``, that let coarse rank 0 - refused
    #: by the travel gate - donate its 3.317 margin to rank 2, which was the
    #: placement the recovery went on to confirm. A margin belonging to a
    #: hypothesis this module has already refused is not evidence for a
    #: different one.
    #:
    #: So the margin is measured again, from the selected hypothesis's own
    #: coverage against the best GENUINELY DIFFERENT rival still standing -
    #: every ranked hypothesis whose coarse placement differs from the selected
    #: one's, except those refinement merged INTO the selected placement, which
    #: are the same answer arrived at twice rather than a rival. A selected
    #: hypothesis that a better-covering different hypothesis outranks fails
    #: this outright, whatever became of that hypothesis under the gates.
    #:
    #: ``None`` for a rival overlap means no genuinely different hypothesis was
    #: standing at all, and the margin is then infinite by construction.
    selected_rank: int | None
    selected_coarse_overlap: float
    selected_rival_overlap: float | None
    selected_overlap_margin: float
    selected_overlap_unambiguous: bool
    #: Origins of the two placements that had to agree, in the order they were
    #: measured: the searching partition first, then each confirming one.
    agreeing: tuple[str, ...]
    #: How far apart the searching placement and each confirmation put the
    #: frame's own surface, in depth units, against the same agreement gate the
    #: ordinary decision uses.
    confirmation_disagreements: tuple[float, ...]
    refusals: Refusals
    work: RecoveryWork
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "triggered_by": self.triggered_by,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "partitions": [partition.as_dict() for partition in self.partitions],
            "best_coarse_overlap": _json_number(self.best_coarse_overlap),
            "rival_coarse_overlap": (
                None
                if self.rival_coarse_overlap is None
                else _json_number(self.rival_coarse_overlap)
            ),
            "coarse_overlap_margin": _json_number(self.coarse_overlap_margin),
            "coarse_overlap_unambiguous": self.coarse_overlap_unambiguous,
            "coarse_score_margin": _json_number(self.coarse_score_margin),
            "coarse_score_unambiguous": self.coarse_score_unambiguous,
            "coarse_limitations": list(self.coarse_limitations),
            "coarse_source_degenerate": self.coarse_source_degenerate,
            "coarse_reference_degenerate": self.coarse_reference_degenerate,
            "hypotheses": [hypothesis.as_dict() for hypothesis in self.hypotheses],
            "selected_rank": self.selected_rank,
            "selected_coarse_overlap": _json_number(self.selected_coarse_overlap),
            "selected_rival_overlap": (
                None
                if self.selected_rival_overlap is None
                else _json_number(self.selected_rival_overlap)
            ),
            "selected_overlap_margin": _json_number(self.selected_overlap_margin),
            "selected_overlap_unambiguous": self.selected_overlap_unambiguous,
            "agreeing": list(self.agreeing),
            "confirmation_disagreements": [
                _json_number(value) for value in self.confirmation_disagreements
            ],
            "refusals": list(self.refusals),
            "work": self.work.as_dict(),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class KeyframeDecision:
    """What happened to the keyframe set because of one accepted frame.

    Recorded rather than inferred. "The set holds frames 0, 14 and 31" does not
    say why frame 22 is absent, and the difference between "it was never novel"
    and "it was evicted as redundant" is the difference between a capture that
    sampled its viewpoints badly and one whose bound is too small.
    """

    frame: int
    action: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {"frame": self.frame, "action": self.action, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class TrackDecision:
    """Where one offered frame went, or why nowhere.

    ``pose`` is never present without ``accepted`` and ``accepted`` is never
    inferred from ``pose``: the two are set together, once, so a caller cannot
    read one and act on the other. ``failure_type`` and :meth:`message` are the
    two fields the live capture hands to
    :class:`~scanner3d.capture_sequence.AlignmentFailure`, so a refusal here
    reaches the durable tracking-loss record with its evidence intact.
    """

    frame: int
    accepted: bool
    pose: Rigid | None
    origin: str | None
    travel: float
    rescue: bool
    candidates: tuple[TrackCandidate, ...]
    keyframes: tuple[int, ...]
    keyframe_decisions: tuple[KeyframeDecision, ...]
    refusals: Refusals
    failure_type: str | None
    #: Origins of the candidates that agreed WITH EACH OTHER and produced this
    #: pose - the selected consensus cluster, and nothing else. A strict subset
    #: of :attr:`passing` whenever a reference cleared its own gates and then
    #: placed the frame somewhere else, which is a thing a reader has to be able
    #: to see: "three references cleared their gates" and "three references said
    #: the same thing" are different statements, and round 1 published the first
    #: under this field's name. For an accepted decision this is never empty and
    #: always contains :attr:`origin` - including the anchor and a seeded bridge
    #: frame, whose one supporting reference is the decision that placed it -
    #: so a reader never has to special-case how support is reported.
    agreeing: tuple[str, ...] = ()
    #: Origins that cleared their gates and were OUTVOTED. Never more than one
    #: for an ordinary placement - see :meth:`IncrementalTracker._agreed`. For a
    #: bounded coarse recovery it is every ordinary reference that cleared its
    #: own gates and then disagreed with the recovered placement, which can be
    #: all of them: a recovery is reached precisely because those references did
    #: not agree with each other, and hiding that they also disagree with the
    #: answer would make the record flatter than the capture was.
    dissenting: tuple[str, ...] = ()
    #: Present when the ordinary decision refused and the bounded coarse
    #: recovery was reached, or when equally large consensus clusters needed
    #: the recovery to arbitrate before the distinctly tighter one could be used
    #: as a fallback. It records both success and refusal. ``None`` on every
    #: unambiguous ordinary placement, which is what makes "the normal path pays
    #: no coarse-search cost" visible in the record rather than only in the
    #: code.
    recovery: RecoveryEvidence | None = None
    limitations: tuple[str, ...] = LIMITATIONS

    @property
    def best(self) -> TrackCandidate | None:
        """The candidate the pose came from, if one did."""
        if not self.accepted or self.origin is None:
            return None
        return next((c for c in self.candidates if c.origin == self.origin), None)

    @property
    def passing(self) -> tuple[str, ...]:
        """Origins of every candidate that cleared its OWN gates, agreement aside.

        Deliberately not called agreeing. Clearing a gate is not agreeing: a
        reference can be finite, converged, inside the residual, matched, used
        and overlapping and still put the frame somewhere the others do not.
        :attr:`agreeing` is the cluster that actually placed it and
        :attr:`dissenting` is the reference that cleared its gates and then
        contradicted them; this is the union those two are drawn from, kept
        because "the model fit fine and disagreed anyway" is evidence about the
        capture that neither of the other two fields alone can express.
        """
        return tuple(candidate.origin for candidate in self.candidates if candidate.accepted)

    def message(self) -> str:
        """One line an operator or a durable record can carry.

        Leads with the first refusal, because that is the one the operator can
        act on, and names the candidates that were tried, because "it did not
        line up" is not actionable and "the model and the previous frame put it
        in two different places" is.
        """
        tried = ", ".join(candidate.origin for candidate in self.candidates) or "no reference"
        recovery = self.recovery
        if self.accepted:
            agreed = ", ".join(self.agreeing) or self.origin or "one reference"
            dissent = (
                f"; {', '.join(self.dissenting)} cleared its gates and disagreed"
                if self.dissenting
                else ""
            )
            if recovery is not None and recovery.accepted:
                # Deliberately a different sentence from an ordinary model
                # placement, because it describes a different event: the
                # ordinary references did not agree, and this frame is here on
                # a coarse search confirmed by two reference sets that share no
                # accepted frame. An operator who reads "placed against the
                # model" for this has been told the scan is going well.
                return (
                    f"recovered by a bounded coarse search: the ordinary references refused "
                    f"({recovery.triggered_by}), and {len(recovery.partitions)} disjoint "
                    f"reference sets agreed on this placement to within "
                    f"{max(recovery.confirmation_disagreements, default=0.0):.3f}"
                    f"{dissent}; travel {self.travel:.3f}. Slow down and keep surface you have "
                    "already scanned in view."
                )
            arbitration = (
                "; a bounded coarse arbitration ran first but produced no independently "
                "confirmed replacement"
                if recovery is not None
                else ""
            )
            return (
                f"placed against the {self.origin} reference; agreed by {agreed}{dissent}; "
                f"travel {self.travel:.3f}{arbitration}"
            )
        first = self.refusals[0] if self.refusals else "no reason was recorded."
        line = f"tracking refused this frame ({tried} tried): {first}"
        if recovery is not None and recovery.attempted and recovery.refusals:
            # Appended rather than leading. The ordinary refusal is the one the
            # operator can act on; that a bounded coarse recovery was also tried
            # and also refused is the answer to "did anything else get a look",
            # which is a question a reader of a tracking-loss record will have.
            line += f" A bounded coarse recovery was tried and refused too: {recovery.refusals[0]}"
        return line

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "frame": self.frame,
            "accepted": self.accepted,
            "origin": self.origin,
            "rescue": self.rescue,
            "pose": self.pose.as_dict() if self.pose is not None else None,
            "travel": _json_number(self.travel),
            "agreeing": list(self.agreeing),
            "passing": list(self.passing),
            "dissenting": list(self.dissenting),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "keyframes": list(self.keyframes),
            "keyframe_decisions": [item.as_dict() for item in self.keyframe_decisions],
            "refusals": list(self.refusals),
            "failure_type": self.failure_type,
            "recovery": None if self.recovery is None else self.recovery.as_dict(),
            "limitations": list(self.limitations),
        }


@dataclass(slots=True)
class _Accepted:
    """One accepted frame as the tracker remembers it.

    ``cloud`` is ``None`` for a frame whose cloud has been released under
    :attr:`TrackingPolicy.max_retained_clouds`. The pose is kept forever - it is
    the answer, and it is small - while the cloud is kept only while it can
    still be a reference.
    """

    index: int
    pose: Rigid
    cloud: tuple[Point, ...] | None


@dataclass(frozen=True, slots=True)
class _Agreement:
    """The ordinary consensus and an optional stronger arbitration to try first.

    ``arbitration_refusal`` is present only for equally large mutual-agreement
    clusters whose tightest member cleared the policy margin. That cluster is
    sufficient ordinary evidence, but a recovery confirmed by two disjoint
    multi-frame references is stronger evidence and gets the first chance to
    place the frame. If it cannot, :attr:`cluster` remains the bounded fallback.
    """

    cluster: tuple[TrackCandidate, ...]
    arbitration_refusal: TrackDecision | None = None


class IncrementalTracker:
    """The accepted model, the keyframes, and the decision that extends them.

    Deterministic and self-contained. Nothing here reads a clock, a device, a
    file or a random number: two trackers offered the same clouds in the same
    order make the same decisions, which is what makes the offline
    reconstruction able to reproduce what the live capture decided.

    Usage is two calls. :meth:`offer` places an ordinary frame. :meth:`seed`
    places one WITHOUT fitting, at a pose the caller already has - which is what
    a recorded relocalization bridge is, and why the offline path does not refit
    across a boundary the capture already decided.
    """

    def __init__(
        self,
        *,
        backend: str = "python",
        policy: TrackingPolicy = DEFAULT_POLICY,
        point_to_plane: bool = True,
        max_correspondence_distance: float = 12.0,
        voxel_size: float = 1.5,
        keep_fraction: float = 0.8,
    ) -> None:
        self._backend = backend
        self._policy = policy
        self._point_to_plane = point_to_plane
        self._max_correspondence_distance = max_correspondence_distance
        self._voxel_size = voxel_size
        self._keep_fraction = keep_fraction
        self._accepted: list[_Accepted] = []
        self._keyframes: list[int] = []
        #: Why each candidate that could not be FORMED could not be, for the
        #: frame currently being decided. Reset at the top of every ``offer``,
        #: so it never describes a frame other than the one being refused.
        self._unformed: list[str] = []
        #: The last accepted SINGLE-STEP relative transform - this module's
        #: only notion of how the scanner is moving, and the whole of it.
        #: ``None`` until two consecutive offers have both been accepted, and
        #: again whenever that continuity breaks: an anchor, a seeded bridge, or
        #: an acceptance with refused frames behind it. A step measured across a
        #: gap of unknown size is not a velocity, and extrapolating one as
        #: though it were a single frame's motion is the failure this whole
        #: mechanism exists to avoid rather than to commit.
        self._velocity: Rigid | None = None
        #: Offers since the last acceptance, INCLUDING the one being decided:
        #: one on the ordinary next frame, two once a frame has been refused.
        #: This is the tracker's only measure of elapsed capture. It exists
        #: because :meth:`offer` numbers frames by ACCEPTANCE, so without it a
        #: refused frame is invisible to everything downstream and the next
        #: prediction would silently under-shoot by a whole step per refusal.
        self._offers_since_accepted = 0
        #: The prediction formed for the frame being offered, or None. Set once
        #: per offer by :meth:`_candidates` and read by :meth:`_judge`, so every
        #: candidate of one frame - recovery hypotheses included - is judged
        #: against the same prediction.
        self._offer_prior: Rigid | None = None
        #: Every ICP solve this tracker has attempted, counted where they are
        #: actually run rather than where they are promised. Read as a
        #: DIFFERENCE around one recovery to fill
        #: :attr:`RecoveryWork.icp_solves`, so the number in the record is what
        #: the solver was asked to do and not what the policy said it might be.
        self._solves = 0

    # -- what a caller may read -------------------------------------------

    @property
    def policy(self) -> TrackingPolicy:
        return self._policy

    @property
    def frames(self) -> int:
        """How many frames have been accepted, which is not how many were offered."""
        return len(self._accepted)

    @property
    def poses(self) -> tuple[Rigid, ...]:
        """Every accepted pose, in acceptance order. The chain this produces."""
        return tuple(item.pose for item in self._accepted)

    @property
    def keyframes(self) -> tuple[int, ...]:
        """Accepted frame indices currently held as keyframes, lowest first."""
        return tuple(self._keyframes)

    @property
    def retained_clouds(self) -> int:
        """Accepted clouds still in memory. Bounded by the policy, always."""
        return sum(1 for item in self._accepted if item.cloud is not None)

    @property
    def retained_points(self) -> int:
        """Points held across every retained cloud.

        The memory bound stated as a number rather than as two numbers a reader
        has to multiply. Never more than
        ``max_retained_clouds * retained_cloud_points``, whatever the caller
        offered and however long the capture ran - which is a claim round 1 made
        and did not keep, and is therefore counted here so a test can check it.
        """
        return sum(len(item.cloud) for item in self._accepted if item.cloud is not None)

    def record(self) -> dict[str, object]:
        """The tracker's own state and policy, for a durable record."""
        return {
            "schema_version": SCHEMA_VERSION,
            "policy": self._policy.as_dict(),
            "backend": self._backend,
            "point_to_plane": self._point_to_plane,
            "frames": self.frames,
            "keyframes": list(self._keyframes),
            "retained_clouds": self.retained_clouds,
            "retained_points": self.retained_points,
            "max_retained_points": (
                self._policy.max_retained_clouds * self._policy.retained_cloud_points
            ),
            "limitations": list(LIMITATIONS),
        }

    # -- placing a frame ---------------------------------------------------

    def seed(self, cloud: Sequence[Point], pose: Rigid, *, anchor: bool = True) -> TrackDecision:
        """Accept one frame at a pose decided elsewhere, without fitting it.

        This is how a recorded relocalization bridge enters the chain, and how
        the frames a recovery published continue one. The boundary was decided
        during the capture, against evidence in its own record, and refitting it
        here would be doing the one thing the boundary exists to prevent:
        handing ICP two frames that are not a small step apart and believing
        what it returns.

        ``anchor`` says whether this frame is the boundary ITSELF, which always
        becomes a keyframe because it is where the model gains a view it has no
        other way of reaching. The frames published behind it are placed by the
        same recorded decision but are ordinary members of the chain, so they
        are offered to the ordinary keyframe rules instead - otherwise one
        published batch of a dozen frames would fill the whole keyframe set with
        one viewpoint.

        Refuses an unusable cloud rather than seeding a frame nothing can be
        placed against afterwards.
        """
        index = len(self._accepted)
        points = tuple(cloud)
        if len(points) < self._policy.min_track_points:
            return self._refused(
                index,
                TRACK_FRAME_EMPTY_TYPE,
                [
                    f"the frame offered as a {BRIDGE_ORIGIN} carries {len(points)} points, under "
                    f"the {self._policy.min_track_points} a placement can be built on."
                ],
            )
        # A seeded frame is never fitted, so no ICP gate stands between it and
        # the accepted model. Everything a later frame is judged against - the
        # model reference, an overlap, a keyframe distance - is built from these
        # points and this pose, so a non-finite one is checked HERE or not at
        # all, and the frame that would finally report a NaN is not this one.
        if not _finite_cloud(points):
            return self._refused(
                index,
                TRACK_FRAME_UNUSABLE_TYPE,
                [
                    f"the frame offered as a {BRIDGE_ORIGIN} carries entries that are not "
                    "three finite coordinates. It is not fitted here, so nothing downstream "
                    "would catch them; seeding it would put values no reference can be measured "
                    "against into the accepted model."
                ],
            )
        if not _rigid_finite(pose):
            return self._refused(
                index,
                TRACK_FRAME_UNUSABLE_TYPE,
                [
                    f"the pose offered for this {BRIDGE_ORIGIN} frame is not a finite rigid "
                    "transform, so every pose composed from it afterwards would be meaningless."
                ],
            )
        placed = bool(self._accepted)
        # 1.0, not 0.0: a seeded frame was not measured against the model
        # reference, so it has no overlap evidence and must not be treated as
        # though it had shown something new. Motion alone decides whether it
        # becomes a keyframe.
        decisions = self._admit(index, points, pose, overlap=1.0, anchor=anchor or not placed)
        return TrackDecision(
            frame=index,
            accepted=True,
            pose=pose,
            origin=BRIDGE_ORIGIN if placed else ANCHOR_ORIGIN,
            agreeing=(BRIDGE_ORIGIN if placed else ANCHOR_ORIGIN,),
            travel=0.0,
            rescue=False,
            candidates=(),
            keyframes=self.keyframes,
            keyframe_decisions=decisions,
            refusals=(),
            failure_type=None,
        )

    def offer(
        self, cloud: Sequence[Point], *, adjacent_fit: LocalFit | None = None
    ) -> TrackDecision:
        """Decide where one ordinary frame goes. NEVER raises for ordinary data.

        ``adjacent_fit`` lets a caller that has ALREADY solved this frame
        against its predecessor hand that solve in rather than paying for it
        twice - the offline reconstruction fits every adjacent pair up front,
        across processes, because those pairs are independent of each other even
        though the poses they compose into are not. The fit must be the fit of
        this cloud onto the immediately previous accepted cloud; everything
        downstream of it is identical either way, which is what keeps the live
        and offline answers the same.

        A refusal mutates NOTHING: no pose, no keyframe, no retained cloud.
        """
        # BEFORE the cloud is even inspected. A frame refused for carrying no
        # geometry is still a frame that arrived and still time the scanner
        # spent moving, so it counts as a delivery step exactly as a frame
        # refused by a gate does. This is the one counter a refusal is allowed
        # to move, and it moves nothing a placement is measured against.
        self._offers_since_accepted += 1
        self._offer_prior = None
        index = len(self._accepted)
        points = tuple(cloud)
        if len(points) < self._policy.min_track_points:
            return self._refused(
                index,
                TRACK_FRAME_EMPTY_TYPE,
                [
                    f"the frame carries {len(points)} usable points, under the "
                    f"{self._policy.min_track_points} a placement can be built on. Either the "
                    "object has left the view or the depth frame is empty; there is nothing "
                    "here to align."
                ],
            )
        if not _finite_cloud(points):
            return self._refused(
                index,
                TRACK_FRAME_UNUSABLE_TYPE,
                [
                    "the frame carries entries that are not three finite coordinates, so "
                    "there is no geometry here to align and nothing a residual measured against "
                    "it would mean. The first accepted frame is not fitted at all, so this is "
                    "checked before anything is admitted rather than left to a solver."
                ],
            )
        if not self._accepted:
            decisions = self._admit(index, points, Rigid.identity(), overlap=0.0, anchor=True)
            return TrackDecision(
                frame=index,
                accepted=True,
                pose=Rigid.identity(),
                origin=ANCHOR_ORIGIN,
                agreeing=(ANCHOR_ORIGIN,),
                travel=0.0,
                rescue=False,
                candidates=(),
                keyframes=self.keyframes,
                keyframe_decisions=decisions,
                refusals=(),
                failure_type=None,
            )

        previous = self._accepted[-1]
        self._unformed = []
        source = _sampled(points, self._policy.candidate_source_points)
        candidates = self._candidates(index, source, previous, adjacent_fit)
        return self._decide(index, points, source, previous, candidates)

    # -- forming candidates ------------------------------------------------

    def _candidates(
        self,
        index: int,
        source: tuple[Point, ...],
        previous: _Accepted,
        adjacent_fit: LocalFit | None,
    ) -> tuple[TrackCandidate, ...]:
        """Up to :attr:`TrackingPolicy.max_candidates` independent placements.

        ``source`` is ONE bounded sample of the offered frame, and every
        candidate - the adjacent one included - is fitted from it, travelled by
        it and compared against the others on it. That is what makes
        :attr:`TrackingPolicy.candidate_source_points` a bound rather than a
        description, and it also means the candidates are several references'
        opinions about the same geometry rather than about different subsets.

        Order matters and is stated: the adjacent candidate is formed FIRST,
        because when it produces a finite transform it is the best available
        prediction of where this frame is, and every other candidate is then a
        refinement seeded from that prediction rather than from "wherever the
        previous frame was". Seeding the keyframe and model solves from a
        stale-by-one-frame pose is what would make them fail on exactly the
        moving-camera capture they exist for.
        """
        candidates: list[TrackCandidate] = []
        prior = self._motion_prior(source)
        self._offer_prior = prior
        adjacent = self._adjacent_candidate(index, source, previous, adjacent_fit, prior)
        if adjacent is not None:
            candidates.append(adjacent)

        # The prediction: the adjacent answer when it is a real transform, the
        # constant-velocity extrapolation when there is one, and the previous
        # pose when there is neither. Note "finite", not "accepted" - a fit that
        # failed a gate can still be a serviceable starting point, and a
        # starting point is not a placement.
        #
        # The middle term is what the previous arrangement gave away. Falling
        # back to ``previous.pose`` is a prediction that the scanner did not
        # move at all, which on a moving capture is the one thing it is known
        # not to have done - and it was the seed for the keyframe and model
        # solves on exactly the frames where the adjacent solve had already
        # failed, which is to say on the frames that needed the best seed
        # available rather than the worst.
        predicted = previous.pose if prior is None else previous.pose.compose(prior)
        if adjacent is not None and adjacent.edge.accepted:
            predicted = adjacent.pose

        confirming: set[int] = {previous.index}
        if self._policy.use_keyframes:
            for keyframe in self._selected_keyframes(previous, predicted):
                confirming.add(keyframe.index)
                candidate = self._candidate(
                    index,
                    source,
                    origin=KEYFRAME_ORIGIN,
                    reference=keyframe.cloud or (),
                    reference_frames=(keyframe.index,),
                    into_world=keyframe.pose,
                    initial=keyframe.pose.inverse().compose(predicted),
                    previous=previous,
                    seed=PREDICTED_SEED,
                )
                if candidate is not None:
                    candidates.append(candidate)
        if self._policy.use_model:
            # The model reference is built from accepted frames NO other
            # candidate is using. Round 1 called the keyframe and model
            # candidates independent while the model reference contained the
            # selected keyframe's own points, so a frame that agreed with itself
            # counted as two references agreeing - at the highest preference,
            # and as one of the two the disagreement gate is supposed to be
            # measuring between.
            reference, frames = self._model_reference(
                previous, predicted, exclude=frozenset(confirming)
            )
            # Two accepted viewpoints at least, or this is not a model
            # reference. Early in a capture the only frame available to it is
            # the anchor, which is also the only keyframe available - and a
            # "model" candidate that is a second copy of the keyframe candidate
            # is not independent evidence, it is one measurement counted twice
            # and given the highest preference for being called something else.
            if len(frames) >= MIN_MODEL_REFERENCE_FRAMES and reference:
                candidate = self._candidate(
                    index,
                    source,
                    origin=MODEL_ORIGIN,
                    reference=reference,
                    reference_frames=frames,
                    into_world=Rigid.identity(),
                    initial=predicted,
                    previous=previous,
                    seed=PREDICTED_SEED,
                )
                if candidate is not None:
                    candidates.append(candidate)
        return tuple(candidates)

    def _motion_prior(self, source: tuple[Point, ...]) -> Rigid | None:
        """Where the last accepted motion says this frame is, or ``None``.

        A body-frame constant-velocity prediction: the step between the last two
        accepted frames, composed once for every delivery step since the last
        acceptance. It is a STARTING POSE and never a placement. Every gate
        downstream of it is the gate an identity-seeded solve meets, unchanged,
        so a prior that seeds a refinement into a wrong local minimum is refused
        by the same residual, overlap, travel and agreement evidence as anything
        else - and a prior that is simply absent costs a frame nothing it was
        not already paying.

        ``None`` wherever the answer would be a guess rather than an
        extrapolation:

        - no velocity has been measured, or the last one was discarded because
          the chain it came from was not continuous;
        - the scanner has been unplaceable for more than
          :attr:`TrackingPolicy.max_prior_steps` delivery steps, past which the
          extrapolation is multiplying an error rather than carrying a
          measurement;
        - the extrapolation already puts the surface beyond
          :data:`MAX_PRIOR_TRAVEL_MULTIPLE` travel gates, where it is a runaway
          rather than a velocity. It may exceed ONE gate: the travel gate is
          judged against the prediction where there is one, so a steady fast
          step is a step and not a jump;
        - the extrapolation is inside the STATIONARY BAND, where the scanner
          counts as not having moved. Then this is not a different starting
          point from the identity the first adjacent attempt is already seeded
          from - by contract, including the fit the offline path hands in - so
          a retry would re-run that solve and reach that answer again. A frame
          the references refuse while the scanner is standing still is refused;
          spending a second solve to say so twice is not a recovery.

        Both bounds are the module's own existing numbers rather than new ones.
        "Far enough to be somewhere else on the object" is
        :attr:`~TrackingPolicy.max_travel_multiple`, already answered once for
        the travel gate; "near enough to count as not having moved" is
        :data:`STATIONARY_DISTANCE_MULTIPLE`, already answered once for keyframe
        selection. A prediction inside the lower bound is also a prediction that
        would have left ``predicted`` where it already was, so nothing else
        loses anything by its absence.
        """
        velocity = self._velocity
        if velocity is None or not self._policy.motion_prior:
            return None
        steps = self._offers_since_accepted
        if steps < 1 or steps > self._policy.max_prior_steps:
            return None
        try:
            prior = velocity
            for _ in range(steps - 1):
                prior = prior.compose(velocity)
        except ScannerError:
            # Composition refuses a non-finite result rather than returning one.
            # An accepted pose is finite, so this is unreachable by ordinary
            # data - and a prediction that cannot be formed is simply absent,
            # which every caller already handles.
            return None
        travel = frame_travel(source, prior)
        if travel > MAX_PRIOR_TRAVEL_MULTIPLE * self._policy.max_travel_multiple * NOISE_FLOOR:
            return None
        if travel <= STATIONARY_DISTANCE_MULTIPLE * NOISE_FLOOR:
            return None
        return prior

    def _adjacent_candidate(
        self,
        index: int,
        source: tuple[Point, ...],
        previous: _Accepted,
        adjacent_fit: LocalFit | None,
        prior: Rigid | None,
    ) -> TrackCandidate | None:
        """The immediately previous accepted frame as a reference.

        ``source`` is the same bounded sample every other candidate is fitted
        from, and the reference is the previous frame's RETAINED cloud, which is
        bounded as it is admitted. A caller supplying ``adjacent_fit`` must have
        fitted those same two clouds - see :func:`bounded_source` and
        :func:`retained_cloud`, which exist so the offline path can build them
        without a tracker.

        Two attempts at most, and the second one only where the first produced
        no ACCEPTED candidate.

        The first is seeded from the identity - the scanner has not moved since
        the last accepted frame - and that ordering is load-bearing rather than
        historical. ``reconstruct_sweep`` pre-fits every adjacent pair up front
        and across processes, where no pose chain exists yet to extrapolate a
        velocity from; a tracker that seeded its FIRST adjacent solve from a
        prediction could no longer accept those pre-fits, and the offline path
        would have to become sequential to keep agreeing with the live one.
        Keeping the identity attempt first is what keeps a run with workers, a
        run without, and the capture that produced the frames on the identical
        chain.

        The retry is seeded from ``prior`` and judged by this same method
        against the same reference and the same gates, so it can only turn a
        refusal into a placement. It never turns one placement into a different
        one: where the first attempt was accepted, the retry does not run and no
        solve is spent. Which seed reached the returned candidate is recorded on
        it as :attr:`TrackCandidate.seed`.
        """
        reference = previous.cloud
        if reference is None:
            # Only reachable if the retention bound is set below two, which
            # ``TrackingPolicy`` forbids - but stated rather than assumed.
            return None

        def judged(fit: LocalFit, seed: str) -> TrackCandidate:
            return self._judge(
                index,
                origin=ADJACENT_ORIGIN,
                fit=fit,
                source=source,
                reference=reference,
                reference_frames=(previous.index,),
                pose=previous.pose.compose(fit.transform),
                previous=previous,
                seed=seed,
            )

        fit = adjacent_fit
        if fit is None:
            fit = self._fit(source, reference, Rigid.identity())
        first = None if fit is None else judged(fit, IDENTITY_SEED)
        if first is not None and first.accepted:
            return first
        if prior is None:
            return first
        retry = self._fit(source, reference, prior)
        if retry is None:
            return first
        seeded = judged(retry, PRIOR_SEED)
        # The seeded attempt replaces the first one by being ACCEPTED where it
        # was not, or by existing where there was no first attempt at all. Two
        # attempts that both failed are reported as the first one, because that
        # is the solve every other reader of this capture - the offline path
        # included - would have run, and a refusal has to name the fit that
        # everybody else will see.
        if seeded.accepted or first is None:
            return seeded
        return first

    def _candidate(
        self,
        index: int,
        source: tuple[Point, ...],
        *,
        origin: str,
        reference: Sequence[Point],
        reference_frames: tuple[int, ...],
        into_world: Rigid,
        initial: Rigid,
        previous: _Accepted,
        seed: str,
        coarse_to_fine: bool = False,
    ) -> TrackCandidate | None:
        """One refinement against one reference, expressed as a world pose.

        ``into_world`` is the whole of the transform-direction bookkeeping.
        :func:`~scanner3d.icp_plane.icp_point_to_plane` returns the transform
        carrying SOURCE coordinates into TARGET coordinates, so:

        - a keyframe target is that keyframe's own cloud in its own frame, and
          the world pose is ``keyframe_pose.compose(fit)``;
        - the model target is already in world coordinates, so ``into_world`` is
          the identity and the fit IS the world pose.

        ``initial`` is stated in the target's frame for the same reason and is
        the caller's prediction carried there.
        """
        if len(reference) < self._policy.min_track_points or not source:
            return None
        if coarse_to_fine:
            # The wide pass, and only for a coarse recovery. Its answer is a
            # STARTING POSE: it is not judged, not recorded as a placement and
            # not allowed to be one. See
            # :data:`RECOVERY_COARSE_DISTANCE_MULTIPLE`.
            wide = self._fit(
                source,
                reference,
                initial,
                distance=(
                    self._max_correspondence_distance
                    * self._policy.recovery_coarse_distance_multiple
                ),
            )
            if wide is None:
                return None
            initial = wide.transform
        fit = self._fit(source, reference, initial)
        if fit is None:
            return None
        return self._judge(
            index,
            origin=origin,
            fit=fit,
            source=source,
            reference=reference,
            reference_frames=reference_frames,
            pose=into_world.compose(fit.transform),
            previous=previous,
            seed=seed,
        )

    def _fit(
        self,
        source: Sequence[Point],
        target: Sequence[Point],
        initial: Rigid,
        *,
        distance: float | None = None,
    ) -> LocalFit | None:
        """One ICP solve, or ``None`` when it refused to run.

        ICP raising is an ordinary outcome for a starting pose that put the
        frame somewhere with nothing under it - too few correspondences survive
        the gate - and it is information about that candidate rather than a
        failure of the frame. A candidate that could not be formed simply is not
        offered to the decision, and WHY it could not be is put in
        ``self._unformed`` so the refusal can say so: "no candidate could be
        formed" without the solver's own words is exactly the unanswerable
        report this whole task exists to stop producing.
        """
        tolerance = convergence_tolerance(target, fraction=self._policy.convergence_noise_fraction)
        # ``distance`` is the ONE thing a caller may vary, and only the wide
        # first pass of a coarse recovery varies it - see
        # :data:`RECOVERY_COARSE_DISTANCE_MULTIPLE`. It changes how far a solve
        # may reach for a correspondence, which is a property of the SEARCH; it
        # changes no gate, and the fit that reaches ``_judge`` is always the one
        # run at the tracker's own distance.
        reach = self._max_correspondence_distance if distance is None else distance
        self._solves += 1
        try:
            if self._point_to_plane:
                return icp_point_to_plane(
                    source,
                    target,
                    initial=initial,
                    tolerance=tolerance,
                    max_correspondence_distance=reach,
                    voxel_size=self._voxel_size,
                    keep_fraction=self._keep_fraction,
                    backend=self._backend,
                )
            return icp(
                source,
                target,
                initial=initial,
                tolerance=tolerance,
                max_correspondence_distance=reach,
                voxel_size=self._voxel_size,
                keep_fraction=self._keep_fraction,
                backend=self._backend,
            )
        except ScannerError as exc:
            self._unformed.append(f"{type(exc).__name__}: {exc}")
            return None
        except (ValueError, ArithmeticError) as exc:
            self._unformed.append(f"{type(exc).__name__}: {exc}")
            return None

    def _judge(
        self,
        index: int,
        *,
        origin: str,
        fit: LocalFit,
        source: Sequence[Point],
        reference: Sequence[Point],
        reference_frames: tuple[int, ...],
        pose: Rigid,
        previous: _Accepted,
        seed: str,
    ) -> TrackCandidate:
        """Every gate one candidate has to clear, with the measurement beside it.

        The first four come from :func:`~scanner3d.relocalize.judge_local_edge`,
        unchanged and deliberately: finiteness, convergence, residual against
        the noise floor, matched and used fractions are the SAME decision the
        tentative local chain makes, and inventing a second looser vocabulary
        for ordinary capture is exactly how a gate stops meaning anything.

        Overlap and travel are added here because this decision needs two things
        a pairwise link does not: whether the frame is actually ON the reference
        it was scored against, and whether the placement it implies is a
        physically plausible step rather than a jump.
        """
        # Measured once, honestly, and used exactly as measured. There is no
        # second, softer reading of this judgement: an accepted candidate is one
        # whose edge cleared every gate ``judge_local_edge`` names, convergence
        # included. The convergence threshold the solver was RUN at is the one
        # this module chose - see :data:`CONVERGENCE_NOISE_FRACTION` - so that
        # gate now refuses fits that never settled rather than fits whose
        # residual stopped mattering four decimal places ago.
        edge = judge_local_edge(fit, frame=index)
        refusals: list[str] = list(edge.refusals)
        failed_gates: list[str] = []
        tolerance = OVERLAP_TOLERANCE_MULTIPLE * NOISE_FLOOR
        overlap = overlap_fraction(
            source,
            reference,
            fit.transform,
            tolerance=tolerance,
            backend=fit.resolved_backend,
        )
        relative = previous.pose.inverse().compose(pose)
        travel = frame_travel(source, relative) if edge.accepted else math.inf
        # How far the placement sits from where the last accepted motion said
        # the frame would be. A scanner carried steadily moves the surface a
        # long way between accepted frames and almost none from its own
        # prediction; a solve that locked onto somewhere else on the object is
        # far from both. The gate below refuses only the second.
        prior = self._offer_prior
        deviation: float | None = None
        if edge.accepted and prior is not None:
            try:
                deviation = frame_travel(source, prior.inverse().compose(relative))
            except ScannerError:
                deviation = None
        gate = self._policy.max_travel_multiple * NOISE_FLOOR
        if edge.accepted:
            if origin != ADJACENT_ORIGIN and overlap < self._policy.min_reference_overlap:
                failed_gates.append(OVERLAP_GATE)
                refusals.append(
                    f"the frame overlaps the {origin} reference by {overlap:.2f}, under the "
                    f"{self._policy.min_reference_overlap} needed for its residual to describe "
                    "the frame rather than a corner of it."
                )
            if not math.isfinite(travel):
                failed_gates.append(TRAVEL_GATE)
                refusals.append(
                    f"the travel implied by the {origin} placement is not a finite distance, so "
                    "there is no step here to judge."
                )
            elif travel > gate and (deviation is None or deviation > gate):
                failed_gates.append(TRAVEL_GATE)
                if deviation is None:
                    refusals.append(
                        f"the {origin} placement moves the surface {travel:.3f} since the last "
                        f"accepted frame, over the {gate:.3f} gate "
                        f"({self._policy.max_travel_multiple:g}x the {NOISE_FLOOR} noise "
                        "floor), and no accepted motion predicts a step that size. That is a "
                        "jump to somewhere else on the object, not a step."
                    )
                else:
                    refusals.append(
                        f"the {origin} placement moves the surface {travel:.3f} since the last "
                        f"accepted frame and {deviation:.3f} from where the last accepted "
                        f"motion predicted it, both over the {gate:.3f} gate "
                        f"({self._policy.max_travel_multiple:g}x the {NOISE_FLOOR} noise "
                        "floor). That is a jump to somewhere else on the object, not a step."
                    )
        return TrackCandidate(
            origin=origin,
            reference_frames=reference_frames,
            reference_points=len(reference),
            source_points=len(source),
            pose=pose,
            edge=edge,
            overlap=overlap,
            travel=travel,
            accepted=not refusals,
            refusals=tuple(refusals),
            failed_gates=tuple(failed_gates),
            seed=seed,
            travel_from_prediction=deviation,
        )

    # -- choosing between them ---------------------------------------------

    def _decide(
        self,
        index: int,
        points: tuple[Point, ...],
        source: tuple[Point, ...],
        previous: _Accepted,
        candidates: tuple[TrackCandidate, ...],
    ) -> TrackDecision:
        """Which candidate, if any, becomes the pose.

        Three ordinary outcomes and no fourth. Nothing survived: refused.
        Survivors that disagree: refused, WITHOUT choosing the smaller residual,
        because a residual cannot arbitrate between two placements that are
        simply not the same answer. Survivors that agree: the one fitted against
        the broadest reference, because agreement has already settled which
        answer it is and what remains is which reference drifts least.

        Each of those three refusals is then offered, and only then, to
        :meth:`_maybe_recover`. A tied pair of equal-size consensus clusters
        whose tightness margin leaves one bounded fallback is offered there too:
        disjoint multi-frame confirmation is stronger evidence, so it gets the
        first chance, but the tighter ordinary cluster remains usable if that
        recovery refuses. An unambiguous ordinary placement never reaches a
        coarse search, and every other frame reaches at most one.
        """
        survivors = [candidate for candidate in candidates if candidate.accepted]
        unformed = tuple(self._unformed)
        if not survivors:
            tried = ", ".join(candidate.origin for candidate in candidates) or "none"
            reasons = [
                f"{candidate.origin}: {candidate.refusals[0]}"
                for candidate in candidates
                if candidate.refusals
            ]
            # The solver's own words for every candidate that never became one,
            # kept because "no candidate could be formed" tells a reader
            # nothing, and this is the message that reaches the durable
            # tracking-loss record.
            reasons.extend(f"a reference could not be fitted at all ({note})" for note in unformed)
            return self._maybe_recover(
                index,
                points,
                source,
                previous,
                candidates,
                self._refused(
                    index,
                    TRACK_NO_CANDIDATE_TYPE,
                    [
                        f"no reference placed this frame ({tried} tried). "
                        + (" ".join(reasons) if reasons else "No candidate could even be formed.")
                    ],
                    candidates=candidates,
                ),
            )

        gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
        agreement = self._agreed(index, survivors, source, gate, candidates)
        if isinstance(agreement, TrackDecision):
            return self._maybe_recover(index, points, source, previous, candidates, agreement)
        cluster = agreement.cluster
        arbitration: TrackDecision | None = None
        if agreement.arbitration_refusal is not None:
            arbitration = self._maybe_recover(
                index,
                points,
                source,
                previous,
                candidates,
                agreement.arbitration_refusal,
            )
            if arbitration.accepted:
                return arbitration
        chosen = min(
            cluster,
            key=lambda candidate: (PREFERENCE_ORDER.index(candidate.origin), candidate.rmse),
        )

        # A MODEL-ONLY survivor is a rescue. Always, and whatever became of the
        # other candidates.
        #
        # Round 1 derived "could anything have confirmed this" from
        # ``len(candidates) > 1`` and so exempted precisely the case the rescue
        # gates exist for: when the adjacent solve RAISED, or the keyframe cloud
        # was released, or the capture was young enough that no second reference
        # had been built yet, the model candidate was the only one FORMED and
        # sailed through with no rescue check at all. A singular flat panel is
        # the textbook cause of that raise. The reason the other candidates are
        # absent is not evidence about this one; the only thing that is evidence
        # is the reference behind it, so the gates are applied on their own
        # terms every time.
        #
        # There is no start-of-capture exemption to remove: the anchor is placed
        # by ``offer`` before any candidate is formed, and a model candidate
        # cannot exist before MIN_MODEL_REFERENCE_FRAMES accepted frames do.
        agreeing = tuple(candidate.origin for candidate in cluster)
        dissenting = tuple(candidate.origin for candidate in survivors if candidate not in cluster)
        rescue = len(survivors) == 1 and chosen.origin == MODEL_ORIGIN
        if rescue:
            refusals = self._rescue_refusals(chosen)
            if refusals:
                return self._maybe_recover(
                    index,
                    points,
                    source,
                    previous,
                    candidates,
                    self._refused(
                        index,
                        TRACK_RESCUE_UNSUPPORTED_TYPE,
                        refusals,
                        candidates=candidates,
                    ),
                )

        decisions = self._admit(index, points, chosen.pose, overlap=chosen.overlap, anchor=False)
        return TrackDecision(
            frame=index,
            accepted=True,
            pose=chosen.pose,
            origin=chosen.origin,
            travel=chosen.travel,
            rescue=rescue,
            candidates=candidates,
            keyframes=self.keyframes,
            keyframe_decisions=decisions,
            refusals=(),
            failure_type=None,
            agreeing=agreeing,
            dissenting=dissenting,
            recovery=None if arbitration is None else arbitration.recovery,
        )

    def _agreed(
        self,
        index: int,
        survivors: list[TrackCandidate],
        source: tuple[Point, ...],
        gate: float,
        candidates: tuple[TrackCandidate, ...],
    ) -> _Agreement | TrackDecision:
        """The one cluster of references that agree, or the refusal that says why not.

        Three things have to hold before a set of independent references counts
        as having agreed on WHERE this frame is, and each of them is a thing
        round 1 got wrong or did not ask:

        1. **The agreement is mutual.** Every pair inside the cluster is within
           ``gate``. Round 1 measured every survivor against the preferred one,
           so two extremes nearly twice the gate apart passed as long as each sat
           near the third.
        2. **The agreement is unambiguous.** Either there is exactly ONE largest
           such cluster, or one of the equally large clusters is distinctly
           tighter than the runner-up by the policy's ambiguity margin. The
           latter is a fallback only: disjoint multi-frame recovery gets the
           first chance because it is stronger evidence. Tightness is the
           cluster's widest pairwise pose disagreement, measured on this frame's
           surface; an ICP residual never arbitrates between placements.
        3. **The dissent is one reference at most.** A single reference
           contradicting a cluster of others is refused when it is HALF the
           evidence and outvoted when it is a minority of it - which with the
           shared policy's three candidates means: two agreeing and one
           dissenting places the frame, one agreeing and one dissenting does not.

        Rule 3 is the one worth arguing, so here is the argument and the
        measurement behind it. This module exists because the adjacent pair is
        the reference that goes wrong first and quietly - and a refusal makes it
        worse in a specific, mechanical way: the adjacent candidate is seeded
        from the identity relative to the last ACCEPTED frame, so after a refused
        frame it is a Gauss-Newton refinement started a whole skipped step away
        from the answer, which is exactly the condition
        :mod:`scanner3d.icp_plane` says its linearisation does not survive.
        Measured on the owner's 2026-08-28 capture, frame 10: the adjacent
        candidate cleared every gate and sat 5.59 and 5.54 from the keyframe and
        model candidates, which sat 1.75 from each other. Letting the one
        reference that had been seeded badly veto the two that had not ended the
        capture there - and ending it there cascaded, because the next frame's
        adjacent reference was then two skipped steps stale, then three, and the
        matched fraction collapsed for the rest of the capture.

        Giving one dissenter a veto is therefore not the conservative reading it
        looks like: it converts one bad seed into a terminal loss. Requiring a
        MAJORITY of independent references to agree is the conservative reading,
        and it is still strictly stronger than round 1, which accepted a set no
        two members of which had been compared with each other.
        """
        clusters = _agreement_clusters(survivors, source, gate)
        largest = len(clusters[0]) if clusters else 0
        dissenting = len(survivors) - largest
        cluster_widths = sorted(
            (
                max(
                    (
                        _disagreement(a.pose, b.pose, source)
                        for a, b in itertools.combinations(cluster, 2)
                    ),
                    default=0.0,
                ),
                position,
                cluster,
            )
            for position, cluster in enumerate(clusters)
        )
        selected: tuple[TrackCandidate, ...] | None = None
        cluster_margin = 1.0
        if largest >= 2 and dissenting <= 1:
            if len(cluster_widths) == 1:
                selected = cluster_widths[0][2]
            elif cluster_widths:
                best_width, _, best_cluster = cluster_widths[0]
                runner_width = cluster_widths[1][0]
                if best_width == 0.0:
                    cluster_margin = math.inf if runner_width > 0.0 else 1.0
                else:
                    cluster_margin = runner_width / best_width
                if cluster_margin >= self._policy.agreement_cluster_margin:
                    selected = best_cluster
        pairs = _disagreements(survivors, source, gate)
        named = "; ".join(f"{a.origin} and {b.origin} ({apart:.3f} apart)" for a, b, apart in pairs)

        def refused(why: str) -> TrackDecision:
            return self._refused(
                index,
                TRACK_DISAGREEMENT_TYPE,
                [
                    f"{len(survivors)} independent references each placed this frame "
                    f"acceptably and then disagreed about where it went - "
                    f"{named or 'no pair was within the gate'} - against the {gate:.3f} gate "
                    f"({AGREEMENT_RESIDUAL_MULTIPLE:g}x the {NOISE_FLOOR} noise floor), and "
                    f"{why}. Choosing the smaller residual would manufacture confidence that "
                    "is not there. Move back over surface already scanned, or slow down."
                ],
                candidates=candidates,
            )

        if selected is not None:
            if len(cluster_widths) > 1:
                best_width = cluster_widths[0][0]
                runner_width = cluster_widths[1][0]
                why = (
                    f"they form {len(clusters)} equally large groups that agree inside "
                    "themselves and not with each other. The tightest group spans "
                    f"{best_width:.3f}, and the next spans {runner_width:.3f}: "
                    f"{cluster_margin:.2f}x apart against the "
                    f"{self._policy.agreement_cluster_margin:.2f}x ambiguity margin. The "
                    "tighter group is the bounded ordinary fallback, but a placement "
                    "confirmed by disjoint multi-frame references is stronger and is tried "
                    "first"
                )
                return _Agreement(selected, refused(why))
            return _Agreement(selected)
        if len(survivors) == 1:
            return _Agreement(tuple(survivors))
        if len(clusters) > 1:
            best_width = cluster_widths[0][0]
            runner_width = cluster_widths[1][0]
            why = (
                f"they fall into {len(clusters)} equally large groups that agree inside "
                "themselves and not with each other. The tightest group spans "
                f"{best_width:.3f}, and the next spans {runner_width:.3f}: only "
                f"{cluster_margin:.2f}x apart against the "
                f"{self._policy.agreement_cluster_margin:.2f}x ambiguity margin, so there is "
                "more than one answer here"
            )
        elif dissenting > 1:
            why = (
                f"{dissenting} of the {len(survivors)} references stand outside the largest "
                "agreement, so no majority of independent references says the same thing"
            )
        else:
            why = "no two of them agree with each other at all"
        return refused(why)

    # -- the bounded coarse recovery ---------------------------------------

    def _maybe_recover(
        self,
        index: int,
        points: tuple[Point, ...],
        source: tuple[Point, ...],
        previous: _Accepted,
        candidates: tuple[TrackCandidate, ...],
        refusal: TrackDecision,
    ) -> TrackDecision:
        """The ordinary refusal, or the placement a bounded coarse recovery found instead.

        Reached from three ordinary refusals - nothing survived, the survivors
        disagreed, or a model-only rescue was unsupported - and from one
        provisional disagreement: tied equal-size ordinary consensus groups
        where one is distinctly tighter. An empty or malformed frame never gets
        here because it is refused in :meth:`offer` before a candidate exists.
        Neither does an unambiguous ordinary placement.

        Whatever this returns, the ORDINARY refusal is preserved when the
        recovery does not place a refused frame: same ``failure_type``, same
        ``refusals``, same absence of mutation. That is what keeps the live
        capture's consecutive-refusal streak, its durable tracking-loss record
        and the post-loss relocalization state machine behaving exactly as they
        did before this fallback existed. The provisional disagreement is the
        one exception: if recovery refuses, its distinctly tighter cluster is
        still admitted by :meth:`_decide`, with the failed recovery evidence
        attached. A recovery can stop a real refusal from happening; it cannot
        turn one that happened into something else.
        """
        if not self._policy.recovery_enabled:
            return refusal
        formed = [candidate for candidate in candidates if candidate.edge.accepted]
        if formed and all(candidate.failed_gates == (TRAVEL_GATE,) for candidate in formed):
            # Every reference that cleared the shared alignment vocabulary
            # cleared it, overlapped its reference, and then put this frame
            # further from the last accepted pose than a step. That is evidence
            # about WHERE the frame is, and a better starting pose cannot change
            # it: a recovered placement is measured for travel against the same
            # last accepted pose and would be refused by the same gate. Spending
            # a coarse search to reach that conclusion again is work, not
            # evidence.
            return replace(
                refusal,
                recovery=self._recovery_declined(
                    refusal,
                    f"every one of the {len(formed)} references that aligned this frame at all "
                    "placed it further from the last accepted pose than a step, and failed no "
                    "other gate. A recovered placement is measured against that same pose by "
                    "that same travel gate, so a coarse search could not reach a different "
                    "answer. Move back over surface you have already scanned.",
                ),
            )
        chosen, recovered, evidence = self._recover(index, source, previous, refusal)
        if chosen is None:
            return replace(refusal, recovery=evidence)
        gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
        dissenting = tuple(
            candidate.origin
            for candidate in candidates
            if candidate.accepted and _disagreement(chosen.pose, candidate.pose, source) > gate
        )
        decisions = self._admit(index, points, chosen.pose, overlap=chosen.overlap, anchor=False)
        return TrackDecision(
            frame=index,
            accepted=True,
            pose=chosen.pose,
            origin=RECOVERY_ORIGIN,
            travel=chosen.travel,
            rescue=False,
            candidates=candidates + recovered,
            keyframes=self.keyframes,
            keyframe_decisions=decisions,
            refusals=(),
            failure_type=None,
            agreeing=evidence.agreeing,
            dissenting=dissenting,
            recovery=evidence,
        )

    def _recovery_declined(self, refusal: TrackDecision, reason: str) -> RecoveryEvidence:
        """Evidence for a refusal the recovery was deliberately not offered.

        Not the same value as an attempt that ran and refused, and that
        difference is the point: a reader has to be able to tell "the evidence
        already answered this" from "four hypotheses were ranked and none of
        them held up", and a single ``accepted: false`` cannot say which.
        """
        return RecoveryEvidence(
            schema_version=RECOVERY_SCHEMA_VERSION,
            triggered_by=refusal.failure_type,
            attempted=False,
            accepted=False,
            partitions=(),
            best_coarse_overlap=0.0,
            rival_coarse_overlap=None,
            coarse_overlap_margin=math.inf,
            coarse_overlap_unambiguous=False,
            coarse_score_margin=math.inf,
            coarse_score_unambiguous=False,
            coarse_limitations=(),
            coarse_source_degenerate=False,
            coarse_reference_degenerate=False,
            hypotheses=(),
            selected_rank=None,
            selected_coarse_overlap=0.0,
            selected_rival_overlap=None,
            selected_overlap_margin=math.inf,
            selected_overlap_unambiguous=False,
            agreeing=(),
            confirmation_disagreements=(),
            refusals=(reason,),
            work=RecoveryWork(
                coarse_source_points=0,
                refine_source_points=0,
                coarse_searches=0,
                coarse_orientations_ranked=0,
                coarse_hypotheses_scored=0,
                hypotheses_refined=0,
                confirmations_refined=0,
                icp_solves=0,
                partitions=0,
                reference_frames=0,
                reference_points=0,
                coarse_reference_points=0,
            ),
            limitations=RECOVERY_LIMITATIONS,
        )

    def _recover(
        self,
        index: int,
        source: tuple[Point, ...],
        previous: _Accepted,
        refusal: TrackDecision,
    ) -> tuple[TrackCandidate | None, tuple[TrackCandidate, ...], RecoveryEvidence]:
        """One bounded coarse-to-fine attempt at the frame the references refused.

        The whole of it, in order, and every step of it bounded by a number the
        policy states and the record carries:

        1. Build :attr:`TrackingPolicy.recovery_partitions` reference sets from
           accepted frames, sharing no accepted frame with each other and none
           of them containing the immediately previous accepted frame - whose
           prediction is what the ordinary decision just failed on.
           Refuse a partition that spans fewer accepted VIEWPOINTS than
           :data:`RECOVERY_MIN_PARTITION_POSE_SPAN`, measured with the tracker's
           own keyframe pose-gap vocabulary rather than counted in indices.
        2. Rank whole-object hypotheses ONCE, with
           :func:`~scanner3d.multiview.coarse_align`, against the first
           partition, offering it the last accepted pose as a prior - see
           :data:`RECOVERY_PRIORS`. One search, the same one
           :mod:`scanner3d.relocalize` uses; there is no second orientation
           search in this repository and this does not add one.
        3. Refuse unless that ranking is unambiguous, TWICE OVER: unless
           :func:`~scanner3d.multiview.coarse_align`'s own verdict says so on
           its own quantity, and unless the best hypothesis's COVERAGE clears
           :data:`RECOVERY_AMBIGUITY_MARGIN` over the best genuinely different
           one. Both before any refinement, so nothing a solver returns
           afterwards can talk the decision out of them.
        4. Refine the best few hypotheses against that partition and judge each
           one with the SAME gates every ordinary candidate is judged with.
        5. Refuse unless the survivors are one placement. Several seeds
           collapsing into one pose is refinement doing its job; two survivors
           that put the frame in two places are two answers, and this module
           does not arbitrate those by residual.
        6. Measure the coverage margin AGAIN, from the hypothesis that was
           actually selected against every genuinely different placement still
           standing - including the higher-ranked ones a gate refused, which
           therefore refuse the recovery rather than lending it their margin.
        7. Refine that one placement against every REMAINING partition and
           require each to clear its own gates and to agree, against the same
           agreement gate the ordinary decision uses.

        Returns the placement, the candidates that agreed on it, and the
        evidence - and on every refusal, ``None`` and the evidence explaining
        which of those six steps stopped it.
        """
        policy = self._policy
        gate = AGREEMENT_RESIDUAL_MULTIPLE * NOISE_FLOOR
        tolerance = OVERLAP_TOLERANCE_MULTIPLE * NOISE_FLOOR
        started = self._solves
        recovery_source = _sampled(source, policy.recovery_source_points)
        partitions = self._recovery_partitions(previous)

        hypotheses: list[RecoveryHypothesis] = []
        survivors: list[TrackCandidate] = []
        survivor_ranks: list[int] = []
        agreeing: list[str] = []
        disagreements: list[float] = []
        best_overlap = 0.0
        rival_overlap: float | None = None
        margin = math.inf
        unambiguous = False
        score_margin = math.inf
        score_unambiguous = False
        selected_rank: int | None = None
        selected_overlap = 0.0
        selected_rival: float | None = None
        selected_margin = math.inf
        selected_unambiguous = False
        coarse_limits: tuple[str, ...] = ()
        source_degenerate = False
        reference_degenerate = False
        orientations_ranked = 0
        scored_count = 0
        confirmations_refined = 0
        coarse_reference_points = 0
        coarse_searches = 0

        def done(accepted: bool, reasons: Sequence[str]) -> RecoveryEvidence:
            return RecoveryEvidence(
                schema_version=RECOVERY_SCHEMA_VERSION,
                triggered_by=refusal.failure_type,
                attempted=True,
                accepted=accepted,
                partitions=tuple(partition for partition, _ in partitions),
                best_coarse_overlap=best_overlap,
                rival_coarse_overlap=rival_overlap,
                coarse_overlap_margin=margin,
                coarse_overlap_unambiguous=unambiguous,
                coarse_score_margin=score_margin,
                coarse_score_unambiguous=score_unambiguous,
                coarse_limitations=coarse_limits,
                coarse_source_degenerate=source_degenerate,
                coarse_reference_degenerate=reference_degenerate,
                hypotheses=tuple(hypotheses),
                selected_rank=selected_rank,
                selected_coarse_overlap=selected_overlap,
                selected_rival_overlap=selected_rival,
                selected_overlap_margin=selected_margin,
                selected_overlap_unambiguous=selected_unambiguous,
                agreeing=tuple(agreeing),
                confirmation_disagreements=tuple(disagreements),
                refusals=tuple(reasons),
                work=RecoveryWork(
                    coarse_source_points=len(recovery_source),
                    refine_source_points=len(source) if hypotheses else 0,
                    coarse_searches=coarse_searches,
                    coarse_orientations_ranked=orientations_ranked,
                    coarse_hypotheses_scored=scored_count,
                    hypotheses_refined=len(hypotheses),
                    confirmations_refined=confirmations_refined,
                    icp_solves=self._solves - started,
                    partitions=len(partitions),
                    reference_frames=sum(len(partition.frames) for partition, _ in partitions),
                    reference_points=sum(partition.points for partition, _ in partitions),
                    coarse_reference_points=coarse_reference_points,
                ),
                limitations=RECOVERY_LIMITATIONS,
            )

        if len(partitions) < policy.recovery_partitions:
            return (
                None,
                (),
                done(
                    False,
                    [
                        f"a bounded coarse recovery needs {policy.recovery_partitions} reference "
                        "sets that share no accepted frame and each span at least "
                        f"{policy.recovery_min_partition_frames} accepted viewpoints. This "
                        f"capture could build {len(partitions)} of them from the "
                        f"{self.retained_clouds} accepted cloud(s) it still holds, leaving out "
                        "the previous accepted frame whose own prediction the ordinary decision "
                        "has just failed on. One reference set cannot confirm itself."
                    ],
                ),
            )

        thin = [
            partition
            for partition, _ in partitions
            if len(partition.frames) < policy.rescue_min_reference_frames
        ]
        if thin:
            # A recovered placement has NO ordinary reference agreeing with it -
            # that is what being reached from a refusal means - so it is the
            # same no-second-opinion placement the model-only rescue gates
            # exist for, and it inherits them whole rather than getting its own
            # weaker vocabulary. Checked here as well as on every candidate
            # below because a partition too thin to rescue anything cannot
            # become thick enough later in this attempt, and finding that out
            # before the coarse search is the difference between refusing and
            # refusing expensively.
            return (
                None,
                (),
                done(
                    False,
                    [
                        "a recovered placement has no ordinary reference agreeing with it, so it "
                        "has to clear the same evidence a model-only rescue clears, and "
                        f"{len(thin)} of the {len(partitions)} recovery reference set(s) span "
                        f"only {sorted(len(partition.frames) for partition in thin)} accepted "
                        f"frame(s), under the {policy.rescue_min_reference_frames} that gate "
                        "needs. A reference assembled from one or two nearby viewpoints is blind "
                        "in the same direction the frame is, and a better starting pose does not "
                        "make it see."
                    ],
                ),
            )

        narrow = [
            partition
            for partition, _ in partitions
            if partition.pose_span < policy.recovery_min_partition_pose_span
        ]
        if narrow:
            # Distinct accepted frames, measured rather than counted. Four
            # indices captured from one standstill are one viewpoint, and a
            # partition of one viewpoint confirms nothing a second reading of
            # the first partition would not have confirmed - it is blind in the
            # same direction the offered frame is. The frame-count gate above
            # cannot see this case, because it counts frames.
            return (
                None,
                (),
                done(
                    False,
                    [
                        "a recovery reference has to span accepted VIEWPOINTS, not merely "
                        f"accepted frame numbers, and {len(narrow)} of the {len(partitions)} "
                        "reference set(s) built for this frame span only "
                        f"{[round(partition.pose_span, 3) for partition in narrow]} of the "
                        f"{policy.recovery_min_partition_pose_span} keyframe motion gate they "
                        "have to reach. Frames captured from one standstill are one view under "
                        "several indices, and two of them cannot confirm each other. Move "
                        "around the object rather than holding still."
                    ],
                ),
            )

        flat = [partition for partition, cloud in partitions if spanned_dimensions(cloud) < 3]
        if flat or spanned_dimensions(recovery_source) < 3:
            # Planar or singular evidence, refused by name rather than left to
            # produce a confident-looking fit. A cloud that does not span three
            # dimensions determines no orientation, so a shape-level ranking
            # over it is ranking noise, and an ICP started from that ranking
            # settles just as tidily as one started from a real pose.
            return (
                None,
                (),
                done(
                    False,
                    [
                        "a bounded coarse recovery ranks whole-object ORIENTATIONS, and "
                        + (
                            "the frame itself does not span three dimensions"
                            if not flat
                            else "the recovery reference built from accepted frames "
                            f"{[list(partition.frames) for partition in flat]} does not span "
                            "three dimensions"
                        )
                        + ". A flat or singular cloud determines no orientation, so there is "
                        "nothing here for a coarse search to rank and no starting pose it could "
                        "honestly produce. Move so the scanner sees some of the object's depth."
                    ],
                ),
            )

        primary, primary_points = partitions[0]
        coarse_reference_points = len(primary_points)
        coarse_searches = 1
        try:
            coarse = coarse_align(
                recovery_source,
                primary_points,
                priors=(previous.pose,),
            )
        except ScannerError as exc:
            # A cloud with too few points, or one that does not span three
            # dimensions - a frame of one flat panel is the ordinary case.
            # Neither determines an orientation, so there is nothing to rank.
            return (
                None,
                (),
                done(
                    False,
                    [
                        f"no coarse orientation could be ranked for this frame against a "
                        f"recovery reference: {exc}"
                    ],
                ),
            )
        source_degenerate = coarse.source_principal_degenerate
        reference_degenerate = coarse.target_principal_degenerate
        score_margin = coarse.margin
        score_unambiguous = coarse.unambiguous
        coarse_limits = coarse.limitations
        # Near-equal covariance extents make the PCA axes a poor SEED; they do
        # not establish that the shape itself has two acceptable placements.
        # ``coarse_align`` deliberately ranks its octahedral alternatives and
        # the explicit last-pose prior even in this case, reports the warning in
        # ``coarse_limits``, and supplies its own ambiguity verdict. Substituting
        # this one bit for that measured verdict was both redundant and wrong on
        # physical capture 20260831-202514: the prior ranked 1.72x clear, whole-
        # frame coverage ranked it 2.41x clear, and two disjoint references
        # refined it only 0.370 apart. The flag remains in the durable evidence;
        # every measured verdict and confirmation below still has to pass.

        ranked = self._rank_coarse(recovery_source, primary_points, coarse, tolerance)
        orientations_ranked = len(coarse.candidates)
        scored_count = len(ranked)
        if not ranked or ranked[0][0] <= 0.0:
            return (
                None,
                (),
                done(
                    False,
                    [
                        f"none of the {scored_count} coarse orientation(s) scored put any of "
                        "this frame on the recovery reference, so there is no starting pose "
                        "here to refine and nothing an overlap would be measuring."
                    ],
                ),
            )
        best_overlap = ranked[0][0]
        rival = next(
            (
                entry
                for entry in ranked[1:]
                if _disagreement(ranked[0][1].transform, entry[1].transform, source) > gate
            ),
            None,
        )
        if rival is not None:
            rival_overlap = rival[0]
            margin = math.inf if rival_overlap <= 0.0 else best_overlap / rival_overlap
        unambiguous = margin >= policy.recovery_ambiguity_margin
        if not score_unambiguous:
            # The SHARED verdict, required rather than recorded and worked
            # around. :func:`~scanner3d.multiview.coarse_align` ranks the whole
            # orientation set - including whatever priors were offered to it -
            # and says whether its own best is clearly best. Round 1 accepted
            # while this was false and substituted the coverage margin below;
            # that substitution is withdrawn. Both are asked.
            #
            # Asked AFTER the coverage ranking, deliberately, so that a refusal
            # on either verdict carries BOTH numbers. The two are measured on
            # different quantities and a reader has to be able to see where they
            # disagreed - which they do, and which is the measured cost of
            # requiring the coarser of the two.
            return (
                None,
                (),
                done(
                    False,
                    [
                        "the shared coarse search ranked its best orientation only "
                        f"{score_margin:.2f}x ahead of the runner-up, under the "
                        f"{AMBIGUITY_MARGIN} margin it judges its own ranking by, so it does not "
                        "call this ranking unambiguous: " + " ".join(coarse_limits)
                    ],
                ),
            )

        if not unambiguous:
            # First and unconditionally, and never withdrawn by anything a
            # solver returns afterwards. Every hypothesis below descends from
            # this one ranking; a residual between two of them says where the
            # optimiser stopped, not which placement is right.
            return (
                None,
                (),
                done(
                    False,
                    [
                        f"the coarse search covered the recovery reference {best_overlap:.2f} at "
                        f"its best orientation and {rival_overlap:.2f} at a genuinely different "
                        f"one - a {margin:.2f}x margin, under the "
                        f"{policy.recovery_ambiguity_margin}. This shape sits several ways on "
                        "the surface already scanned and the geometry does not say which. "
                        "Refining an ambiguous start picks one of them; it cannot tell them "
                        "apart. Move back over a part of the object you have already scanned "
                        "that does not look like the rest of it, and slow down."
                    ],
                ),
            )

        for rank, (overlap, hypothesis) in enumerate(ranked[: policy.recovery_hypotheses]):
            unformed_before = len(self._unformed)
            refined = self._candidate(
                index,
                source,
                origin=RECOVERY_ORIGIN,
                reference=primary_points,
                reference_frames=primary.frames,
                into_world=Rigid.identity(),
                initial=hypothesis.transform,
                previous=previous,
                seed=COARSE_SEED,
                coarse_to_fine=True,
            )
            refined = self._rescue_gated(refined)
            hypotheses.append(
                RecoveryHypothesis(
                    rank=rank,
                    coarse_origin=hypothesis.origin,
                    coarse_overlap=overlap,
                    coarse_score=hypothesis.score,
                    initial=hypothesis.transform,
                    refined=None if refined is None else refined.pose,
                    accepted=refined is not None and refined.accepted,
                    refusals=(
                        tuple(
                            f"the solver refused this hypothesis ({note})"
                            for note in self._unformed[unformed_before:]
                        )
                        if refined is None
                        else refined.refusals
                    ),
                )
            )
            if refined is not None and refined.accepted:
                survivors.append(refined)
                survivor_ranks.append(rank)

        if not survivors:
            return (
                None,
                (),
                done(
                    False,
                    [
                        f"the coarse search ranked one clear best orientation, and none of the "
                        f"{len(hypotheses)} hypotheses refined from it cleared the same gates an "
                        "ordinary placement clears: "
                        + " ".join(
                            f"rank {item.rank}: {item.refusals[0]}"
                            for item in hypotheses
                            if item.refusals
                        )
                    ],
                ),
            )

        clusters = _agreement_clusters(survivors, source, gate)
        if len(clusters) != 1 or len(clusters[0]) != len(survivors):
            apart = _disagreements(survivors, source, gate)
            named = "; ".join(f"{value:.3f} apart" for _a, _b, value in apart)
            return (
                None,
                (),
                done(
                    False,
                    [
                        f"{len(survivors)} of the refined hypotheses each cleared every gate and "
                        f"then placed this frame in different places ({named or 'no pair agreed'}"
                        f") against the {gate:.3f} agreement gate. Refinement may collapse "
                        "several starting poses into one placement; where it does not, these are "
                        "two answers, and the smaller residual is not evidence for either."
                    ],
                ),
            )

        chosen = survivors[0]
        selected_rank = survivor_ranks[0]
        selected_overlap, selected_candidate = ranked[selected_rank]
        # Every ranked hypothesis that is a DIFFERENT PLACEMENT from the
        # selected one and is still standing as an alternative to it. A
        # hypothesis refinement carried INTO the selected placement is not an
        # alternative - that is two seeds reaching one answer, which the
        # contract explicitly allows and which a margin must not punish. Every
        # other genuinely different hypothesis counts, INCLUDING the ones the
        # gates refused: this margin is about whether the shape distinguishes
        # the placements, and "the solver could not fit that one" is not the
        # shape saying anything. In particular a higher-covering hypothesis that
        # failed travel makes the margin less than one and refuses the recovery,
        # which is exactly the round-1 defect - rank 0 failing and rank 2
        # inheriting its confidence - closed.
        merged = {
            item.rank
            for item in hypotheses
            if item.refined is not None and _disagreement(chosen.pose, item.refined, source) <= gate
        }
        rivals = [
            overlap
            for position, (overlap, candidate) in enumerate(ranked)
            if position != selected_rank
            and position not in merged
            and _disagreement(selected_candidate.transform, candidate.transform, source) > gate
        ]
        if rivals:
            selected_rival = max(rivals)
            selected_margin = (
                math.inf if selected_rival <= 0.0 else selected_overlap / selected_rival
            )
        selected_unambiguous = selected_margin >= policy.recovery_ambiguity_margin
        if not selected_unambiguous:
            return (
                None,
                (),
                done(
                    False,
                    [
                        f"the hypothesis this recovery actually selected - coarse rank "
                        f"{selected_rank} - covers the recovery reference "
                        f"{selected_overlap:.2f}, against {selected_rival:.2f} for a genuinely "
                        f"different placement still standing beside it: a "
                        f"{selected_margin:.2f}x margin, under the "
                        f"{policy.recovery_ambiguity_margin}. The margin that matters is the "
                        "selected placement's own; a higher-ranked hypothesis this module has "
                        "already refused cannot lend it confidence. Move back over a part of the "
                        "object you have already scanned that does not look like the rest of it, "
                        "and slow down."
                    ],
                ),
            )
        agreeing.append(RECOVERY_ORIGIN)
        confirmations: list[TrackCandidate] = []
        for partition, cloud in partitions[1:]:
            confirmation = self._candidate(
                index,
                source,
                origin=RECOVERY_CONFIRM_ORIGIN,
                reference=cloud,
                reference_frames=partition.frames,
                into_world=Rigid.identity(),
                initial=chosen.pose,
                previous=previous,
                seed=RECOVERED_SEED,
            )
            if confirmation is not None:
                confirmations_refined += 1
            confirmation = self._rescue_gated(confirmation)
            if confirmation is None:
                return (
                    None,
                    (),
                    done(
                        False,
                        [
                            "the placement could not be refined at all against the independent "
                            f"reference built from accepted frames {list(partition.frames)}, so "
                            "nothing confirms it."
                        ],
                    ),
                )
            if not confirmation.accepted:
                return (
                    None,
                    (),
                    done(
                        False,
                        [
                            "the independent reference built from accepted frames "
                            f"{list(partition.frames)} did not confirm this placement: "
                            f"{confirmation.refusals[0]}"
                        ],
                    ),
                )
            apart_here = _disagreement(chosen.pose, confirmation.pose, source)
            disagreements.append(apart_here)
            if apart_here > gate:
                return (
                    None,
                    (),
                    done(
                        False,
                        [
                            "the independent reference built from accepted frames "
                            f"{list(partition.frames)} placed this frame {apart_here:.3f} away "
                            f"from where the searching reference did, over the {gate:.3f} "
                            "agreement gate. Two reference sets that do not agree are two "
                            "answers, not a recovery."
                        ],
                    ),
                )
            confirmations.append(confirmation)
            agreeing.append(RECOVERY_CONFIRM_ORIGIN)
        return chosen, (chosen, *confirmations), done(True, ())

    def _rank_coarse(
        self,
        source: tuple[Point, ...],
        reference: Sequence[Point],
        coarse: CoarseAlignment,
        tolerance: float,
    ) -> list[tuple[float, AlignmentCandidate]]:
        """The coarse hypotheses, best COVERAGE first, bounded and deterministic.

        Re-ranked rather than taken in the order
        :func:`~scanner3d.multiview.coarse_align` returns them, and the reason is
        measured rather than aesthetic. That function ranks a DECIMATED cloud by
        a truncated mean nearest-neighbour distance on a grid whose cell is 4% of
        the reference's extent. This module has a finer instrument available and
        a reason to prefer it: the quantity that decides whether a refined
        placement is admitted at all is
        :func:`~scanner3d.multiview.overlap_fraction` over the WHOLE frame at
        this module's own overlap tolerance - the same tolerance :meth:`_judge`
        measures every candidate's overlap at - so ranking the shortlist by it
        refines the hypotheses most likely to clear the gate they will be
        measured by. Measured on the incompatible-minima fixture: the shared
        ranking's best and this ranking's best are two spellings of one
        placement, both refining to the same pose, and the coverage order is
        0.375/0.276/0.120/0.107 where the shared scores are
        5.13/3.92/9.87/9.69.

        The last accepted pose is in this list too - see :data:`RECOVERY_PRIORS`
        - and is ranked by the same coverage measurement as every generated
        orientation. It routinely wins, because a partial view's centroid is not
        the multi-frame reference's centroid and every generated orientation
        starts by putting them together; that is a property of the search, not a
        privilege of the prior, and a prior that covers the reference worse than
        a generated orientation ranks below it.

        Ties break on the coarse score and then on the order the search returned
        them, so two runs of the same capture rank the same hypotheses the same
        way. Truncated to :attr:`TrackingPolicy.recovery_ranked_hypotheses`
        BEFORE anything is scored, so the coverage measurement is bounded too.
        """
        considered = list(coarse.candidates[: self._policy.recovery_ranked_hypotheses])
        scored = [
            (
                overlap_fraction(source, reference, candidate.transform, tolerance=tolerance),
                position,
                candidate,
            )
            for position, candidate in enumerate(considered)
        ]
        scored.sort(key=lambda entry: (-entry[0], entry[2].score, entry[1]))
        return [(overlap, candidate) for overlap, _position, candidate in scored]

    def _recovery_partitions(
        self, previous: _Accepted
    ) -> tuple[tuple[RecoveryPartition, tuple[Point, ...]], ...]:
        """Bounded reference sets that share no accepted frame with each other.

        Disjoint BY CONSTRUCTION rather than by a check afterwards: the frames
        are chosen, then dealt round-robin into the partitions. Dealing rather
        than splitting is the part that matters - a partition made of the first
        half of the capture and one made of the second half are disjoint and are
        also two different parts of the object, so their agreement would be
        about which of them the frame happens to be near. Dealt alternately,
        each partition spans the same range of viewpoints as the other, which is
        what makes them two views of one model rather than two models.

        Chosen NEAREST THE LAST ACCEPTED POSE rather than spread evenly across
        the capture, for the reason :meth:`_reference_frames` already gives for
        the ordinary model reference: a reference budget spent on viewpoints the
        frame cannot see makes the part of it the frame CAN see sparse, and a
        sparse reference measures its own sparsity rather than the placement.
        Measured on ``20260828-105933`` at ``frame_0032`` with the budget spread
        evenly instead - partitions of accepted frames 0, 8, 16, 24 and 4, 12,
        20, 28, half of them the far side of the chassis - the best coarse
        hypothesis covered 0.19 against the same 0.35 gate the ordinary model
        candidate cleared at 0.47.

        The last accepted pose is the right thing to measure that from and the
        only trusted one available: it is an accepted pose, and it is the same
        pose the travel gate judges the recovered placement against. The
        PREDICTION is not available here, because a prediction that held is
        exactly what the ordinary decision has just failed to produce.

        The immediately previous accepted frame is left out of all of them. It
        is the reference whose prediction the ordinary decision has just failed
        on, and a recovery that confirmed itself against it would be re-asking
        the question that produced the refusal.

        A partition that cannot reach
        :attr:`TrackingPolicy.recovery_min_partition_frames` accepted viewpoints
        is dropped rather than padded, so a capture too young to support a
        recovery produces fewer partitions than the policy asks for and the
        caller refuses - which is the honest outcome, not a smaller confirmation.
        """
        policy = self._policy
        eligible = [
            item
            for item in self._accepted
            if item.cloud is not None and item.index != previous.index
        ]
        want = policy.recovery_partitions * policy.recovery_partition_frames
        if len(eligible) > want:
            nearest = sorted(
                eligible,
                key=lambda item: (self._pose_distance(previous.pose, item.pose), item.index),
            )[:want]
            # Back into capture order before dealing, so alternate viewpoints go
            # to different partitions rather than alternate distances.
            eligible = sorted(nearest, key=lambda item: item.index)
        built: list[tuple[RecoveryPartition, tuple[Point, ...]]] = []
        for offset in range(policy.recovery_partitions):
            group = eligible[offset :: policy.recovery_partitions]
            if len(group) < policy.recovery_min_partition_frames:
                continue
            per_frame = max(1, policy.recovery_partition_points // len(group))
            gathered: list[Point] = []
            used: list[int] = []
            poses: list[Rigid] = []
            for position, item in enumerate(group):
                if not item.cloud:
                    continue
                sample = _sampled(item.cloud, per_frame, offset=position)
                if not sample:
                    continue
                used.append(item.index)
                poses.append(item.pose)
                gathered.extend(item.pose.apply_all(sample))
            cloud = tuple(gathered[: policy.recovery_partition_points])
            if (
                len(used) < policy.recovery_min_partition_frames
                or len(cloud) < policy.min_track_points
            ):
                continue
            # Measured here rather than gated here: the caller refuses on it, so
            # a partition that spans no viewpoint change is still BUILT and still
            # recorded with the span that condemned it. A refusal a reader
            # cannot see the number behind is the report this module exists to
            # stop producing.
            span = max(
                (
                    self._pose_distance(a, b)
                    for position, a in enumerate(poses)
                    for b in poses[position + 1 :]
                ),
                default=0.0,
            )
            built.append(
                (
                    RecoveryPartition(frames=tuple(used), points=len(cloud), pose_span=span),
                    cloud,
                )
            )
        return tuple(built)

    def _rescue_gated(self, candidate: TrackCandidate | None) -> TrackCandidate | None:
        """One recovery candidate, re-judged under the model-only rescue gates.

        Every placement a recovery produces is a placement no ordinary reference
        agreed with - that is what being reached from a refusal means - so it is
        precisely the no-second-opinion case :meth:`_rescue_refusals` exists
        for, and it inherits those gates rather than being given a private
        weaker set. A better STARTING POSE is the only thing a coarse search
        contributes; it contributes no evidence at all about how many accepted
        viewpoints the reference spans or how much of the frame is on it, which
        is what those two gates measure.

        Applied to the confirming partitions as well as the searching one. A
        confirmation against a reference too thin or too little overlapped to
        rescue anything is not a confirmation.
        """
        if candidate is None or not candidate.accepted:
            return candidate
        refusals = self._rescue_refusals(candidate, reference="recovery")
        if not refusals:
            return candidate
        return replace(
            candidate,
            accepted=False,
            refusals=candidate.refusals + tuple(refusals),
        )

    def _rescue_refusals(self, chosen: TrackCandidate, *, reference: str = "model") -> list[str]:
        """Why a MODEL-ONLY placement is not supported enough to be believed.

        Reached whenever the model candidate is the only survivor - whether the
        adjacent pair and the keyframes were refused, could not be formed, or
        were never available. That is precisely the singular-view case this
        module exists for, and precisely the case where a flattering fit is most
        dangerous, because there is no second opinion left to contradict it. So
        the reference itself has to be evidence: several accepted viewpoints,
        and the frame substantially on them.

        Convergence is NOT re-checked here, because it is no longer possible to
        get this far without it: a survivor is a candidate whose
        :attr:`~TrackCandidate.edge` cleared every gate
        :func:`~scanner3d.relocalize.judge_local_edge` names, and convergence is
        one of them. What these two gates add is the evidence a weak view can
        still fake with a settled fit - a reference assembled from one or two
        nearby viewpoints is blind in the same direction the frame is, and a
        small residual over a corner of the frame is not a measurement of the
        frame.
        """
        refusals: list[str] = []
        if len(chosen.reference_frames) < self._policy.rescue_min_reference_frames:
            refusals.append(
                f"the previous frame could not place this one and the {reference} reference is "
                f"built from only {len(chosen.reference_frames)} accepted frame(s), under the "
                f"{self._policy.rescue_min_reference_frames} needed. A reference assembled from "
                "one or two nearby viewpoints is blind in the same direction the frame is, so "
                "it cannot rescue it."
            )
        if chosen.overlap < self._policy.rescue_min_overlap:
            refusals.append(
                f"the previous frame could not place this one and the frame overlaps the "
                f"{reference} reference by only {chosen.overlap:.2f}, under the "
                f"{self._policy.rescue_min_overlap} a placement with no second opinion has to "
                "clear."
            )
        return refusals

    def _refused(
        self,
        index: int,
        failure_type: str,
        refusals: Sequence[str],
        *,
        candidates: tuple[TrackCandidate, ...] = (),
    ) -> TrackDecision:
        """A refusal carrying whatever was measured before it was reached.

        Deliberately reports the keyframe set and NO keyframe decisions: nothing
        was decided about the set, because nothing was accepted.
        """
        return TrackDecision(
            frame=index,
            accepted=False,
            pose=None,
            origin=None,
            travel=math.inf,
            rescue=False,
            candidates=candidates,
            keyframes=self.keyframes,
            keyframe_decisions=(),
            refusals=tuple(refusals),
            failure_type=failure_type,
        )

    # -- extending the accepted model --------------------------------------

    def _admit(
        self,
        index: int,
        points: tuple[Point, ...],
        pose: Rigid,
        *,
        overlap: float,
        anchor: bool,
    ) -> tuple[KeyframeDecision, ...]:
        """Append one accepted frame and update the bounded state around it.

        The ONLY place this class mutates. Everything above it measures and
        decides; a refusal never reaches here, which is what makes "a refusal
        changes nothing" a property of the code rather than a promise.
        """
        # BEFORE the append, because the step being measured is the one from
        # the frame that is currently last to the one arriving.
        #
        # Recorded ONLY for an ordinary frame that arrived with nothing refused
        # behind it. That is the single case in which this step is also the step
        # the next prediction is going to assume: an anchor has no predecessor,
        # a seeded bridge crossed a seam that is not motion at all, and an
        # acceptance with refusals behind it spans a gap whose size the next
        # frame has no way to divide back out. Each of those clears the velocity
        # instead, so the prediction is absent rather than wrong.
        if anchor or not self._accepted or self._offers_since_accepted != 1:
            self._velocity = None
        else:
            self._velocity = self._accepted[-1].pose.inverse().compose(pose)
        self._offers_since_accepted = 0
        # Retained THINNED, not whole. Every reference this module builds comes
        # out of these clouds, so an unbounded retained cloud makes every
        # downstream "bounded" claim untrue - which is exactly what the physical
        # replay measured in round 1.
        self._accepted.append(
            _Accepted(
                index=index,
                pose=pose,
                cloud=_sampled(points, self._policy.retained_cloud_points),
            )
        )
        decisions = list(self._update_keyframes(index, pose, overlap=overlap, anchor=anchor))
        self._release_clouds()
        return tuple(decisions)

    def _update_keyframes(
        self, index: int, pose: Rigid, *, overlap: float, anchor: bool
    ) -> list[KeyframeDecision]:
        """Insert, retain or evict, from motion and novelty evidence only.

        Never from the frame number. A capture that stands still delivers frames
        at the same rate as one that moves, and a set filled by counting frames
        would hold twelve views of one spot exactly when the tracker most needed
        twelve different ones.

        Two independent reasons to insert:

        - **Motion.** The pose sits further from EVERY existing keyframe than
          the motion gate. This is the ordinary case and the one that makes a
          moving camera acquire geometrically distinct references.
        - **Novelty.** The frame overlapped the model reference poorly while
          still having moved out of the stationary band. Rounding a corner moves
          the scanner barely at all and changes what it can see completely, and
          motion alone misses that.

        A frame inside the stationary band is never a keyframe under either
        reason, which is the whole of "a long stationary sequence must not fill
        the keyframe set".
        """
        if anchor or not self._keyframes:
            self._keyframes.append(index)
            return [
                KeyframeDecision(
                    frame=index,
                    action="anchor",
                    reason="the first accepted frame is the anchor of the model frame, so it is "
                    "always a keyframe and is never evicted.",
                )
            ]
        gaps = [self._pose_gap(pose, self._pose_of(keyframe)) for keyframe in self._keyframes]
        nearest_degrees = min(gap[0] for gap in gaps)
        nearest_distance = min(gap[1] for gap in gaps)
        moved = (
            nearest_degrees > KEYFRAME_MIN_DEGREES
            or nearest_distance > KEYFRAME_MIN_DISTANCE_MULTIPLE * NOISE_FLOOR
        )
        stationary = (
            nearest_degrees <= STATIONARY_DEGREES
            and nearest_distance <= STATIONARY_DISTANCE_MULTIPLE * NOISE_FLOOR
        )
        novel = not stationary and overlap < KEYFRAME_NOVEL_OVERLAP
        if not (moved or novel):
            return [
                KeyframeDecision(
                    frame=index,
                    action="retain",
                    reason=(
                        f"this pose sits {nearest_degrees:.2f} degrees and "
                        f"{nearest_distance:.3f} from the nearest of "
                        f"{len(self._keyframes)} keyframes, inside the "
                        f"{KEYFRAME_MIN_DEGREES:g} degree / "
                        f"{KEYFRAME_MIN_DISTANCE_MULTIPLE * NOISE_FLOOR:.3f} motion gate, and "
                        f"its {overlap:.2f} overlap is not novel. The set is unchanged."
                    ),
                )
            ]
        reason = (
            f"this pose is {nearest_degrees:.2f} degrees and {nearest_distance:.3f} from the "
            f"nearest keyframe, past the {KEYFRAME_MIN_DEGREES:g} degree / "
            f"{KEYFRAME_MIN_DISTANCE_MULTIPLE * NOISE_FLOOR:.3f} motion gate."
            if moved
            else (
                f"this pose overlaps the model reference by only {overlap:.2f}, under the "
                f"{KEYFRAME_NOVEL_OVERLAP} novelty gate, having moved out of the stationary "
                "band - a new view from nearly the same place."
            )
        )
        self._keyframes.append(index)
        decisions = [KeyframeDecision(frame=index, action="insert", reason=reason)]
        decisions.extend(self._evict())
        return decisions

    def _evict(self) -> list[KeyframeDecision]:
        """Bring the keyframe set back inside its bound, deterministically.

        The anchor is never evicted, because it defines the model frame. The
        newest :attr:`TrackingPolicy.protected_recent_keyframes` are never
        evicted, because they are where the scanner is now. Among the rest, the
        one whose pose is CLOSEST to another keyframe goes: it is the most
        redundant, so losing it costs the least coverage. Ties go to the lowest
        index, so the answer does not depend on iteration order.
        """
        decisions: list[KeyframeDecision] = []
        while len(self._keyframes) > self._policy.max_keyframes:
            protected = self._policy.protected_recent_keyframes
            evictable = self._keyframes[1 : len(self._keyframes) - protected]
            if not evictable:
                # The bound is smaller than the anchor plus the protected tail,
                # so nothing may go. Stated rather than looping forever.
                decisions.append(
                    KeyframeDecision(
                        frame=self._keyframes[-1],
                        action="retain",
                        reason=(
                            f"the keyframe set holds {len(self._keyframes)} over its "
                            f"{self._policy.max_keyframes} bound, but every entry is either the "
                            "anchor or inside the protected recent tail, so none may be evicted."
                        ),
                    )
                )
                break
            scored = []
            for keyframe in evictable:
                pose = self._pose_of(keyframe)
                nearest = min(
                    self._pose_distance(pose, self._pose_of(other))
                    for other in self._keyframes
                    if other != keyframe
                )
                scored.append((nearest, keyframe))
            _, victim = min(scored)
            self._keyframes.remove(victim)
            decisions.append(
                KeyframeDecision(
                    frame=victim,
                    action="evict",
                    reason=(
                        f"the set was over its {self._policy.max_keyframes} bound and this "
                        "keyframe was the closest of the evictable ones to another keyframe, so "
                        "it carried the least coverage the rest of the set does not already have."
                    ),
                )
            )
        return decisions

    def _release_clouds(self) -> None:
        """Drop the clouds no reference can use, keeping poses intact.

        Keyframes and the most recent frames are what a candidate is ever built
        from, so everything else is memory that can never be read. Poses stay:
        they are the answer, they are three floats and nine, and the caller
        needs the whole chain.
        """
        # The bound counts RETAINED clouds, so an oversized keyframe set cannot
        # smuggle extra clouds past it. What it also must not do is release a
        # cloud the rest of this class assumes is present, so the order is
        # stated and the first two entries are unconditional: the ANCHOR, which
        # defines the model frame and can never be re-derived, and the MOST
        # RECENT accepted frame, which is the next adjacent candidate's whole
        # reference. ``TrackingPolicy`` refuses any budget too small to hold
        # those plus every keyframe, so this loop never has to choose between
        # them; it is written to make the promise legible rather than implied.
        budget = self._policy.max_retained_clouds
        chosen: set[int] = {self._keyframes[0], self._accepted[-1].index}
        for keyframe in reversed(sorted(self._keyframes)):
            if len(chosen) >= budget:
                break
            chosen.add(keyframe)
        for item in reversed(self._accepted):
            if len(chosen) >= budget:
                break
            chosen.add(item.index)
        for item in self._accepted:
            if item.index not in chosen:
                item.cloud = None

    # -- references --------------------------------------------------------

    def _pose_of(self, index: int) -> Rigid:
        return self._accepted[index].pose

    @staticmethod
    @lru_cache(maxsize=POSE_GAP_CACHE_PAIRS)
    def _pose_gap(a: Rigid, b: Rigid) -> tuple[float, float]:
        """Immutable pose pairs recur in selection/eviction; cache exact results."""
        relative = a.inverse().compose(b)
        return (math.degrees(relative.rotation_angle), relative.translation_magnitude)

    @classmethod
    def _pose_distance(cls, a: Rigid, b: Rigid) -> float:
        """One comparable number for how far apart two poses are.

        Degrees and a distance cannot be compared with each other, so each is
        divided by its own keyframe motion gate first and the larger of the two
        ratios is the answer. That makes "far apart" mean the same thing here,
        in keyframe selection and in eviction, and it makes the number unitless
        so a future change to the depth scale does not silently reweight
        rotation against translation.
        """
        degrees, distance = cls._pose_gap(a, b)
        return max(
            degrees / KEYFRAME_MIN_DEGREES,
            distance / (KEYFRAME_MIN_DISTANCE_MULTIPLE * NOISE_FLOOR),
        )

    def _selected_keyframes(self, previous: _Accepted, predicted: Rigid) -> list[_Accepted]:
        """The keyframes worth refining against, nearest predicted pose first.

        The immediately previous accepted frame is excluded even when it is a
        keyframe: a candidate built from it would be the adjacent candidate
        again, and two copies of one measurement are not two independent
        references. A keyframe whose cloud has been released is skipped rather
        than reconstructed - the retention bound is a bound.
        """
        available = [
            self._accepted[keyframe]
            for keyframe in self._keyframes
            if keyframe != previous.index and self._accepted[keyframe].cloud is not None
        ]
        available.sort(key=lambda item: (self._pose_distance(predicted, item.pose), item.index))
        return available[: self._policy.keyframe_candidates]

    def _reference_frames(
        self, previous: _Accepted, predicted: Rigid, *, exclude: frozenset[int] = frozenset()
    ) -> list[int]:
        """Which accepted frames the model reference is assembled from.

        The keyframes, plus a few recent frames so the reference reaches the
        surface the scanner is on RIGHT NOW rather than only the viewpoints it
        thought worth keeping - MINUS every accepted frame another candidate for
        this same offered frame is already using. ``exclude`` carries the
        previous accepted frame and every selected keyframe, so the model
        candidate is fitted against accepted frames no other candidate has seen.
        That is what "independent" has to mean before an agreement between two
        of them is evidence about the placement rather than about one accepted
        frame's own consistency.

        Bounded by :attr:`TrackingPolicy.max_reference_frames`. When there are
        more frames than that, the ones NEAREST the predicted pose are kept.
        Nearest rather than evenly spread across the capture, and that choice is
        measured rather than aesthetic: a reference budget spent on viewpoints
        the frame cannot see makes the part of it the frame CAN see sparse, and
        a sparse reference produces a placement measurably worse than the
        adjacent pair's - which then loses to it, and the model reference stops
        contributing anything. Keyframes are chosen for their spread in the
        first place, so the nearest few of them still span several viewpoints,
        which is what the rescue gate needs.
        """
        blocked = exclude | {previous.index}
        recent = [
            item.index
            for item in self._accepted[-(self._policy.model_recent_frames + 1) :]
            if item.index not in blocked
        ]
        chosen = sorted(
            {
                index
                for index in [*self._keyframes, *recent]
                if index not in blocked and self._accepted[index].cloud is not None
            }
        )
        limit = self._policy.max_reference_frames
        if len(chosen) <= limit:
            return chosen
        by_distance = sorted(
            chosen, key=lambda index: (self._pose_distance(predicted, self._pose_of(index)), index)
        )
        return sorted(by_distance[:limit])

    def _model_reference(
        self, previous: _Accepted, predicted: Rigid, *, exclude: frozenset[int] = frozenset()
    ) -> tuple[tuple[Point, ...], tuple[int, ...]]:
        """A bounded sample of the accepted model, in accepted-model coordinates.

        Points are spread evenly across the selected frames and evenly within
        each, so the result is bounded by
        :attr:`TrackingPolicy.model_reference_points` however long the capture
        has run, and building it costs the same on frame 900 as on frame 9.

        Returns the points and the frames they came from. The frame list is not
        decoration: :data:`RESCUE_MIN_REFERENCE_FRAMES` is measured against its
        length, so "how many viewpoints is this reference" has to travel with
        the reference itself.
        """
        frames = self._reference_frames(previous, predicted, exclude=exclude)
        if not frames:
            return ((), ())
        per_frame = max(1, self._policy.model_reference_points // len(frames))
        points: list[Point] = []
        used: list[int] = []
        for position, index in enumerate(frames):
            item = self._accepted[index]
            if not item.cloud:
                continue
            sample = _sampled(item.cloud, per_frame, offset=position)
            if not sample:
                continue
            used.append(index)
            points.extend(item.pose.apply_all(sample))
        return (tuple(points[: self._policy.model_reference_points]), tuple(used))


def adjacent_only(policy: TrackingPolicy = DEFAULT_POLICY) -> TrackingPolicy:
    """``policy`` with the references a self-chained segment does not have.

    Named rather than constructed inline so the one caller that needs it - the
    tentative segment of a relocalization, which has no accepted model to refer
    to - is visibly running the same gates as everything else with two reference
    sources absent, not a second looser policy.
    """
    return replace(policy, use_keyframes=False, use_model=False, recovery_enabled=False)
