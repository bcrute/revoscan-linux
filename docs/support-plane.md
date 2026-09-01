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
sample; every sum is `math.fsum`, which is correctly rounded and therefore independent of the
order the points arrived in. The same points in any order give bit-identical answers, and a
rigidly transformed cloud gives the same decision and the same kept points.

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
