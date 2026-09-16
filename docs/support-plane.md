# Support-plane suppression

*Status: implemented, tested only on synthetic and rendered scenes. No claim is made about
any physical scan until the owner runs one.*

## What problem it solves

The object turns. The turntable, the bed, the sheet of MDF it stands on does not.

Every frame therefore contains a large, stationary, strongly planar patch, and pairwise
registration has no notion of which points belong to the thing being reconstructed. The
cheapest way for ICP to explain the next frame is to leave the bed where it is, so the
fitted pose comes back near identity. Two things the operator actually sees follow from
that: the model **freezes** while the object is plainly moving, or the fused result is a
**bed-shaped blob** with a smear on top of it.

## What the filter does

`scanner3d.support_plane.suppress_support_plane` fits **one plane per frame** from that
frame's own geometry and removes **only the narrow band of points lying in it**.

It does **not**:

- crop to a box, or discard everything on one side of the plane;
- assume any direction is "up" — there is no gravity reference on this scanner, and the
  support surface is wherever the operator put it, at whatever slant;
- remove background, walls, hands, clamps or a second object;
- fill holes or estimate geometry that was not measured;
- touch the captured frames. The raw `frame_NNNN.raw` bytes are unchanged and complete;
  everything here happens on a derived cloud, and on a derived in-memory frame, between
  reading a capture and registering or fusing it.

## When it acts, and when it deliberately does not

A fitted plane is acted on only when **every** gate agrees it is a *surrounding support
surface*. Any one failing returns the whole cloud unchanged with the reason recorded:

| Gate | What it rejects |
|---|---|
| inlier count and fraction | a plane through a handful of points |
| two-dimensional spread across the frame | a compact planar face inside a bigger scene |
| footprint: the plane must reach beyond the geometry standing on it | the **lid of a box**, which otherwise looks exactly like a bed |
| enough non-plane geometry left over | a frame pointed at a panel, whose one plane *is* the object |
| that geometry predominantly on one side | a plane cutting through the object |
| enough kept, bounded removal | a frame that is almost entirely bed |

**A large flat chassis surface is deliberately preserved on ambiguity.** Losing a bed costs
one frame that still has its bed in it. Removing a body panel costs the scan. The filter is
built to make the first mistake and not the second, so a sweep where some frames withdrew is
the expected conservative outcome, not a fault.

Because the plane is fitted **per frame and independently**, a single sweep will normally
have some frames where suppression applied and some where it withdrew.

## Where it runs

`scanner3d.preparation.prepare_frame` is the one path from a raw depth frame to a
registration cloud, used by the window's capture, tentative relocalization recovery and
Reconstruct paths **and** by `scanner3d reconstruct`. The order is fixed and stated on every
record:

```
deproject -> depth clip -> deterministic thinning -> speckle filtering -> support-plane filtering
```

A frame that caught nothing stops after the thinning and its record says both filters were
*not run* — which is a different fact from either of them having run and removed nothing.

## Two artifacts, filtered separately

A reconstruction produces **two** things from one capture, by two different routes, and the
support plane has to leave both of them:

- the **registered cloud** is built from the decimated per-frame clouds the path above
  produces. Speckle rejection and support-plane suppression both act here, and nothing that
  either removed contributed to a pose.
- the **exported mesh** is fused from the **full-resolution frames** at the accepted poses.
  It does not read the decimated clouds at all, so a plane removed only from those would
  still be in the PLY — the bed-shaped blob, intact, in the file the operator opens.

So for each frame the support filter **applied** to, the same plane and the same band are
masked out of an **in-memory copy** of that frame before it is fused: the masked samples are
set to the sensor's own no-return value, which the fusion already skips and counts. Nothing
else about the frame changes, the fusion's arithmetic is untouched, and
`frame_NNNN.raw` on disk is never written to.

A frame the filter **withdrew** on, and a frame it never ran on, are fused exactly as they
were captured. The speckle filter is **not** applied to the fused frames: it ran on the
decimated cloud, and the full-resolution frames still contain every stray it rejected.

## What you are told

- **While a reconstruction runs**, at a bounded cadence: how many frames the support plane
  was removed from, how many points that was, **how many points were kept**, and how many
  frames kept their whole cloud and why. One further line says how many full-resolution
  frames were masked before fusion.
- **In `reconstruction.json`**, twice over: as prose on the exported cloud's provenance,
  under the `per-frame rejection` limitation, and as a structured schema-versioned aggregate
  under `input.preparation`. That aggregate has one block per artifact —
  `registered_cloud` and `full_resolution_fusion` — so neither one's counts can be read into
  the other's file.
- **In `scanner3d reconstruct --json`**, under `command.preparation` as the same structured
  record and `command.preparation_statement` as the same sentence. That command registers and
  reports without fusing a surface, so its `full_resolution_fusion` block says `ran: false`
  rather than being omitted.

A per-frame record (`SupportPlaneResult.as_dict()`) additionally carries the fitted plane,
the band, the inlier and kept counts and fractions, the spread and footprint measurements,
the one-sided evidence, the number of candidates **actually scored**, the refusal reason, and
the **effective** thresholds — the values the caller supplied, not this module's defaults,
with `overridden_by_caller` naming any that differ from the shipped policy.

## Determinism

No random seed, no RANSAC, no wall clock, no optional dependency, no device state. Candidate
normals are covariance eigenvectors refined by a bounded mode-and-refit loop over a bounded
sample: the window starts an eighth of the cloud's diagonal wide and halves down to the band
that is finally removed, placed at each width where it holds the most points along the current
normal, then centred on the points it fitted once a window holds no more than the last did (see
the 2026-09-11 section below for why). Every sum is `math.fsum`, which is correctly rounded and therefore
independent of the order the points arrived in. The same points in any order give bit-identical
answers, and a rigidly transformed cloud gives the same decision and the same kept points.

## What has and has not been shown

Shown, in `tests/test_support_plane.py` and `tests/test_preparation.py`: exact, slanted and
noisy support planes removed; a chassis plate above a bed keeping every plate point while the
bed goes; withdrawal on a panel, a compact face, a two-sided plane, a box lid, and sparse,
degenerate, non-finite and oversized input; invariance under rigid motion and input order; a
custom threshold deciding a frame and being reported as the value that decided it; typed
refusals for malformed, non-sequence and non-finite already-deprojected points; and a
synthetic rotating object over a dominant bed where adjacent ICP returns identity without
suppression and recovers the known 8-degree motion to within 1.5 degrees with it.

Shown in `tests/test_gui_reconstruct_export.py`: a rendered sphere resting on a slanted bed,
exported through the window's own Reconstruct path, whose mesh has **no vertex in the bed's
band** — while the same clouds, poses and voxel budget fused from the **unmasked** frames put
hundreds of vertices there. That control is the load-bearing half: without it the assertion
would also pass for a field that never reached the bed. The captured frames are byte-identical
before and after, and a sweep of withdrawn frames is fused as the exact bytes that were
captured.

Not shown: anything about a physical scan. The gates' thresholds were chosen against
constructed scenes, and the first real capture may well want them retuned.


## Measured on real captures: it has never applied

Status added 2026-09-02. This filter has now been checked against every kind of
capture this machine has taken, and **it has fired on none of them**.

`docs/tracking.md` already records that it "applied to none of its 176 frames"
on the hand-carried chassis capture `20260828-105933`. That was read at the time
as conservatism doing its job on an ambiguous scene. The turntable case is the
one this filter was written for — a stationary bed under a rotating object, the
exact arrangement in the "What problem it solves" section above — so it should
be where the filter earns its keep.

Sampled across `20260901-201341`, a 190-frame turntable sweep of the
reconstruction target, correctly framed and tracking 190/190:

| frames sampled | applied | refusal reasons |
| --- | --- | --- |
| 8 | **0** | `off-plane-geometry-on-both-sides` (3), `too-few-inliers` (3), `spread-too-narrow` (2) |

`off-plane-geometry-on-both-sides` is the interesting one, and it is most likely
the filter working **correctly** rather than failing: the target's own `base_top`
is a large flat surface, so the dominant plane in these frames is plausibly the
target's face rather than the turntable, and geometry above it (the ramp, the
plateau) and below it (the turntable) is exactly what that refusal describes.
Removing that plane would take the object's own top off, which is the failure
this filter's conservatism exists to prevent — "losing a bed costs one frame,
removing a panel costs the scan".

So this is not necessarily a defect. But it is a fact worth stating plainly:

**Every downstream consumer that assumes the support surface has been removed is
running on clouds that still contain it.** Measured on the sweep above, the
turntable and the room are the majority of the fused cloud: 405,310 points
total, of which the target's own 92 x 82 x 56 mm accounts for a small minority,
and a connected-component search at 4 mm cells returns **one component of
404,379 points spanning 385 mm** — target, turntable and background are one
touching body and no cell size separates them.

That is why `docs/coverage-and-subject.md` finds the coverage gate outvoted
three-to-one by scenery, and why the CAD comparison gate had to be reconstructed
to query from the reference side rather than isolate the subject first.

**Not changed here.** Loosening the gates to make the filter fire is the wrong
repair and would risk exactly the panel removal the thresholds were chosen to
prevent. What this section records is that the filter's real-world hit rate is
zero, so no other module may assume its effect.

## 2026-09-11: a curved object could capture the search (policy version 2)

Found when the reconstruction paths started deprojecting under the unit's metric
correction ([metric-validation.md](metric-validation.md)): the rendered sphere on
a slanted bed that `tests/test_gui_reconstruct_export.py` uses had its bed found
under the factory record and missed under the corrected one - a 0.4% change of
scale. Sweeping the thinning budget showed it was luck either way: the bed, three
quarters of the frame, was missed on 1 of 7 budgets under the factory record and
4 of 7 under the corrected one, and never for a reason the gates state - the
search itself returned a slab through the sphere with a tenth of the points.

**Mechanism.** The seeds are the cloud's covariance eigenvectors, and on that frame
the smallest one is 14 degrees off the bed's normal. The refinement then placed a
window one applied band wide (1% of the diagonal, about 1.5 mm) where it held the
most points along that seed. A bed seen at a grazing angle is long, and a long
plane projected along a direction 14 degrees off its normal is spread over some
60 mm, so a 1.5 mm window catches a ribbon of it at best - while a sphere always
has a patch lying square to the seed, which the same window catches whole. A patch
square to the seed refits to the seed, so every candidate settled on the sphere
and stayed. The deflation rounds removed one sphere patch after another and
re-seeded on what was left, which still pointed at the sphere.

**Change.** The window now starts at an eighth of the diagonal and halves down to
the applied band, and at every width it is placed afresh where it holds the most
points along the refitted normal. A plane is dense along its own normal and a
curved patch is not, so the wide window is claimed by the bed's bulk, the refit
turns towards it, and each narrower window keeps the bed and loses the object. At
the applied band the window goes on being placed by density while each window
holds more points than the last, and only then is the plane centred on the points
it fitted - the version-1 loop from that point on, so the surface it describes is
never left on its own band's edge. No gate moved. Bounded as before: at most eight
halvings, then eight located and eight settling passes, each one linear pass over
a sample of at most 4000 points; the sliding window is evaluated with NumPy where
it is installed, bit for bit the scalar loop's answer (pinned in
`tests/test_live_acceleration.py`). Cost on the real fixture frames, filter alone:
14.6 to 19.6 ms per 900-point live frame and 36 to 40 ms per 2200-point
reconstruction frame, on this machine.

**Shown.** In `tests/test_support_plane.py`, a closed-form grazing bed with a ball
resting on it, off-centre so the covariance seed is 10 to 23 degrees off the bed's
normal: the bed is found as the same 1320 points, to a twentieth of a degree, at
every ball density down to a bed that is half the cloud (the gates then withdraw
on the sparsest ball, correctly, because the bed is more than the removal bound of
the frame). At 40% of the cloud it is not found, and the deflation rounds do not
recover it; that limit is recorded in the test rather than claimed away. The
rendered scene above is found at every thinning budget under both deprojections,
and every scene the version-1 loop did find is found again as the same plane.

**On the 24 real turntable frames** in `tests/fixtures/turntable/`, at the live
and reconstruction budgets and under both deprojections (96 runs): the filter
applied to none of them before and applies to none of them now, as the section
above records. What changed is what it found and why it withdrew: the winning
candidate's population grew in 92 of the 96 runs (median 260 to 434 points), and
the reasons moved from `too-few-inliers` (40 runs to 11) towards
`off-plane-geometry-on-both-sides` (25 to 60). The search is no longer the reason
those frames keep their bed; the gates are, and on those frames the dominant plane
has geometry on both sides of it - the target's own base with the turntable
below and the ramp above, as the section above already argued. Nothing here
changes that judgement.
