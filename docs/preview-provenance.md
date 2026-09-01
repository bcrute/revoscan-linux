# Preview provenance: what is measured, what is derived, and where it is said

## The report this answers

The appliance shows three pictures of one scan, and until this change none of
them said what it was.

- The depth panel is a false-coloured view of the sensor's own 16-bit codes,
  with a neutral checker wherever the sensor returned nothing. Nothing in that
  picture is reconstructed, and nothing in it is a distance.
- The model panel's dots are those returns, deprojected through the factory
  intrinsics and placed by this scan's own pose estimates. They are measurements
  at coordinates that are estimates — and the same panel later holds the
  completed registered reconstruction, so the sentence about them has to be true
  of both.
- The opt-in surface is derived geometry, and there are **two** of them made by
  two different producers. While a capture runs it is a voxel occupancy grid
  marched into triangles: geometry this application invented on cell edges, from
  the poses accepted **as the scan ran**. The owner directly observed it as lumpy
  beside dots that looked clean, at the finest cell edge the live preview can
  build (capture `20260827-152842`: 116 frames, cell 1.981 held from the first
  refresh to the last, 774300 cells, 66616 faces, never coarsened). Once
  Reconstruct has run, the same panel and the same checkbox instead hold the
  surface it exported: the zero level of a truncated signed-distance field fused
  from the captured full-resolution frames at the registration's **corrected**
  poses.

The difference between the three was already written down — in a tooltip on one
checkbox. A tooltip is read by somebody who already suspects there is something
to read. An operator comparing a sparse depth frame, a clean cloud and a lumpy
shell should not have to hover to find out which pixels are sensor returns and
which shape was made up.

The first round of this work made the panel say all of that, and got the last
part wrong in a specific way: it had one wording for "the surface", so the final
reconstruction was described on screen as `DERIVED occupancy surface … from the
poses accepted during capture`. That sentence is false about the one surface the
operator is asked to trust, and it was false in a believable way. Two producers
need two accounts, and something has to carry which is which.

## What changed

**Two permanent lines, one per picture, and no new geometry.** This is a change
to what the appliance SAYS. No sample, pose, registration decision, surface,
field, mesh, export, readiness gate or quality threshold moved.

### Under the depth image

A fixed line states three things: the image is **raw sensor depth codes**, the
grey checker is **no return** rather than a near or far reading, and **no
completion, filling, smoothing or carry-over** is applied. It names no physical
unit, because the raw codes establish none
(`scanner3d.viewer.FrameStats`). It is set once when the panel is built and is
never rewritten, so no frame, run boundary or return-health verdict can take it
away — a scan whose returns are disappearing is exactly when somebody starts
wondering whether the picture is real. The returned/no-return statistics, the
checker rendering, the clipping warning and the return-health line beside it are
unchanged; `docs/return-health.md` still owns those.

### Under the model

One line describes whichever picture is **actually mounted in the viewer**:

| mounted picture | what the line says |
| --------------- | ------------------ |
| nothing yet     | no measured points and no derived surface in this view |
| the dots        | MEASURED depth returns; placement depends on the calibration and on this scan's pose estimates, which are estimates rather than surveyed coordinates; the cloud on the screen is sampled and thinned for display |
| the live shell  | DERIVED occupancy surface — NOT measured; its current triangle count and cell edge; marched from the grid the poses accepted during capture were folded into; can look faceted or lumpy and can stand outside the dots; the coverage percentages and the red colouring describe the POINTS |
| the exported surface | DERIVED signed-distance (TSDF) surface — NOT measured; its own triangle count and voxel size; the zero level of a field fused from the captured full-resolution frames at this reconstruction's CORRECTED poses; no dimension checked against a physical reference; the coverage percentages and the red colouring describe the POINTS |

The points row is one sentence for both clouds the panel can hold — the live
preview and the completed registered reconstruction — so it says what is true of
either and calls neither of them the other.

The wording lives in `scanner3d.preview_provenance`, which is a pure formatter
over one three-state enum for the mounted picture, one two-state enum for WHICH
surface produced the mesh, and one small record carrying that kind with the
surface's triangle count and cell edge. Both enums' members are short
identifiers rather than the sentences, so a rewording is not a change of state
and nothing downstream can end up branching on display text. The record refuses a
kind that is not one of the known producers, a non-integer or non-positive
triangle count and a non-finite or non-positive cell edge; the formatter refuses
a surface with no numbers, a points picture that carries a surface's numbers at
all, and a surface kind with no wording of its own. There is deliberately no
default: any default would invent a claim about pixels, and here the only two
candidates are the two sentences that are false about each other.

### Which surface, and who says so

`SurfaceKind` is set by the producer, beside the mesh it made, and by nothing
else:

- `_live_snapshot` — the live capture path — marks its accumulator's mesh
  `LIVE_OCCUPANCY`;
- `_start_reconstruct` marks `export.surface`, the mesh it just wrote to the
  capture folder, `RECONSTRUCTED_TSDF`.

It travels on `ModelSnapshot` with the geometry and the voxel size, is turned
into the panel's record in the same call that hands the mesh to the viewer, and
is never inferred from the snapshot's note, from whether the view re-frames, from
the checkbox, from the remembered preference or from a default. A published
surface that does not say which it is raises `ProvenanceUnusable` **before** the
mesh is mounted, so an undescribable surface is never on the screen under a
guess.

Three visible claims are driven from that one record, so they cannot contradict
one another: the persistent provenance line, the surface clause of the coverage
summary, and the surface control's own title and tooltip. Round one corrected
only the first, which left the summary calling the exported reconstruction an
"Experimental occupancy preview" built from "poses accepted so far", and left the
checkbox saying it draws the occupancy grid while it was drawing the fused
surface. A control with no surface to describe gets its own wording rather than
either producer's.

### Why it follows the viewer and not the checkbox

The checkbox and the remembered preference say what the operator ASKED for.
Those can disagree with the screen, and the disagreement is normal: a capture
that outgrows the live cell budget has no surface to draw, so the panel answers
with the points while the preference survives for the next surface. The line is
therefore derived from `CloudViewer.showing_surface` and `CloudViewer.has_cloud`
— which forward to whichever backend is mounted — every time anything can change
which picture that is:

- a published model, whose surface and numbers are written in the same call;
- the operator toggling;
- the control following a surface that arrived, was withdrawn, or came back;
- a GPU-to-software fallback, which restores the cloud, the surface, the camera
  and the choice onto the replacement before it reports itself. A failure queued
  by a renderer that has already been swapped out is dropped by the viewer, so
  it cannot relabel the replacement.

Starting a capture withdraws the previous run's shell from the picture before
the new run has anything of its own, so a new scan can never open on a derived
shape whose triangle count and cell edge belong to a scan that has already
finished. The operator's choice is untouched, and the first surface the new
capture builds brings their picture back with it.

## What is deliberately still offline

Three modules in this repository can invent depth samples or weigh invented ones,
and none of them is reached by the GUI, the capture, the tracker, the preview,
the fusion or the export:

- `scanner3d.depth_completion` — spatial completion of small holes
  (`docs/depth-completion.md`);
- `scanner3d.depth_temporal` — temporal carry-over between frames
  (`docs/depth-temporal.md`);
- `scanner3d.tsdf_input` — the weighted fusion input that would let a
  reconstructed sample into a field (`docs/weighted-tsdf-input.md`).

They exist as measured, reproducible offline evidence. The benchmark result is
why they stay there: completion buys completeness that the fused surface does
not need, the weighted-input experiment showed no same-input output gain, and
the shell oracle selected no safe pruning rule
(`docs/depth-completeness-benchmark.md`, `docs/surface-oracle.md`,
`docs/shell-discriminant.md`). Making the preview honest about what is measured
is not a reason to start routing invented samples into what the operator is
shown; if a future task ever displays a completed pixel, it has to arrive with
the same visible provenance boundary and be reviewable on its own.

## Evidence

`tests/test_gui_preview_provenance.py` asks, in this order:

1. the formatter's wording and every refusal at its boundary, without Qt;
2. the real depth panel, built by
   `ScannerWindow._build_depth_panel` on the offscreen Qt platform: the line is
   present, needs no hover, survives frames, warnings and run boundaries, and
   the checker it names is the checker the renderer actually draws;
3. that the presentation work changed no pixel and no statistic — the same
   bytes produce the same false colour and the same `FrameStats` before and
   after the panel draws them, the frame itself is unmodified, and the numeric
   line still reports exactly what was measured;
4. the mounted-picture state through the real window methods: the empty panel,
   the first points snapshot, a surface arriving without being selected, an
   explicit selection, toggling back, a withdrawal with the preference left on,
   a restoration, a new scan, and a checkbox made to disagree with the screen in
   both directions;
5. which surface, end to end and through the real producers: a live capture
   whose published surface is described as the occupancy grid it is, a real
   Reconstruct — frames read, registered, fused and exported by the product —
   whose selected surface is described as the fused signed-distance one it is,
   with the coverage summary and the control agreeing in both cases; a snapshot
   dressed in the other producer's note and re-framing flag, to show the kind is
   not read off either; and a published surface with no kind at all, which is
   refused rather than drawn;
6. a GPU-to-software fallback under both pictures, and a queued failure from the
   renderer that was already replaced;
7. the boundary above, twice: statically, that no module on the way to the
   screen so much as names one of the three adapters, and at runtime, that a
   live capture and a reconstruction both still produce their pictures with
   every public callable in all three adapters replaced by one that raises.
