# Coverage, and the subject problem

Status: research and a diagnosis, not an implementation. Written after a
coverage gate refused a good capture on 2026-09-01, on evidence from this
repository, from the sibling `mapper` repository, and from the literature.
Nothing here has been built. The recommendations at the end are ranked by how
much of the problem they close against how much they cost.

Relevant to both projects. `mapper` reached the same wall first and named it
better; this scanner has machinery `mapper` does not that makes part of it
answerable. Section 6 says what transfers each way.

## 1. What happened

A 190-frame turntable sweep of the reconstruction target tracked perfectly —
every frame placed, held-out loop closure 1.20° — and was then refused:

```
216 of 459 occupied cells of size 18.926 were seen by at least 2 frames
spanning at least 20.0 degrees of view direction; that is 0.471 of the
occupied surface and policy requires at least 0.500
```

Breaking those cells down by distance from the object's centre:

| distance from centre | cells | covered | median observations | median spread |
| --- | --- | --- | --- | --- |
| 0–40 mm | 47 | **96%** | 62 | 121° |
| 40–70 mm | 163 | **72%** | 20 | 64° |
| 70–110 mm | 198 | 43% | 7 | 17° |
| 110–170 mm | 114 | **9%** | 3 | 7° |
| 170+ mm | 4 | 0% | 2 | 2° |

The target is 92 × 82 × 56 mm, so its own surface lies within about 62 mm of
centre — the first two bands, **210 cells at 80% covered**, comfortably past the
gate. Everything beyond is the turntable rim and the room: 316 cells, seen a
handful of times each through a few degrees of arc, failing by construction.

**The metric was outvoted three-to-one by scenery.** It reported 0.471 for a
capture whose subject was at 0.80.

## 2. Why no threshold fixes it

The obvious repair is to stop counting cells with almost no evidence. Sweeping
that bar on the same capture:

| min observations to count as surface | cells | covered | fraction |
| --- | --- | --- | --- |
| 1 (current) | 526 | 259 | 49% |
| 2 | 468 | **259** | 55% |
| 3 | 426 | **259** | 61% |
| 8 | 281 | 239 | 85% |

The numerator does not move between bars 1 and 3 — every excluded cell was
already failing — so it is not shrinking the sample to manufacture a pass, and
it flips to passing at a bar of two.

**It is still wrong.** A genuine facet of the object seen only twice *is* a
coverage gap, and excluding it would hide exactly the failure the gate exists to
catch. The rule cannot tell "scenery I never meant to scan" from "part of my
subject I under-scanned", because both look like a cell with few observations.
Tuning the constant makes the number agreeable on this capture and silently
unsafe on the next one.

The defect is not the threshold. **The metric has no notion of what was being
scanned**, and that is not expressible in the data it is given.

Worse, the direction of the error flips with the capture style:

| | what dilutes the number | effect |
| --- | --- | --- |
| turntable, room in frame | background counter-rotates into a halo of thin cells | **under**-reports |
| handheld, object against a wall | the wall is rigid and well observed | **over**-reports |

So the same gate reads pessimistically in one mode and optimistically in the
other, and no single constant can be right in both.

## 3. What `mapper` already worked out

The sibling repository hit this and named it better. From
`mapper/docs/track-reconstruction.md` §6, the `completeness` object carries
**two verdicts that are never fused**:

| Field | What it answers |
| --- | --- |
| `observed_geometry` | Do the boundaries that *were* reconstructed hang together? |
| `survey_extent` | How much of the course was surveyed. **Always `unknown` — the only member the enum has.** |

And the reasoning, verbatim: *"The mapper cannot know a course extends past what
it saw. There is no prior layout, no expected length, and no closure constraint
to violate — a 55% loop and a genuinely shorter course are the same evidence."*

Two things transfer directly.

**The split.** This scanner's `supported_fraction` is trying to be both verdicts
at once: *"of the surface I reconstructed, how much was seen well"* (answerable)
and *"of the object, how much did I get"* (not answerable from a point cloud
alone). Fusing them is the same defect `mapper` renamed away from.

**The renaming discipline.** `mapper` changed `state` to
`{observed_geometry, survey_extent, missing_segments}` and *renamed rather than
supplemented*, so `completeness["state"]` raises `KeyError` in every reader —
because a live gate "must move by someone's decision rather than by continuing
to compile."

There is a third lesson, and it is uncomfortable. `mapper` records that its
`MIN_OPPOSITE_SIDE_LENGTH_RATIO` *"was called `MIN_COVERAGE_FRACTION` until
MAP-WF-0005 and its docstring claimed it measured coverage of the observed
marker extent. It never has: the denominator is the opposite reconstructed
side's chain length."* A constant whose name asserted one thing while its
denominator computed another. **This repository has the same bug in the same
place**: `coverage.well_covered_fraction` divides by *every occupied cell in the
world frame*, not by the subject.

## 4. What the literature does

The standard representation is not "occupied or not". It is a **four-way voxel
state maintained by ray casting** ([PB-NBV](https://arxiv.org/pdf/2501.10663),
and the next-best-view line generally):

| state | meaning |
| --- | --- |
| **Occupied** | a measurement landed here |
| **Empty** | a ray passed through and hit something further away — *known* to be empty |
| **Unknown** | behind an occupied voxel along every ray so far — never observed |
| **Frontier** | boundary between known and unknown; where new information is |

The update rule is the whole idea: *along a ray, everything before the first
occupied voxel becomes Empty, and everything behind it becomes Unknown.*

That distinction — **empty because I looked, versus unknown because I could
not** — is precisely what this repository's coverage module lacks, and it is
what makes the difference between "surface I failed to cover" and "there is
nothing there".

For bounded scenes there is a further trick.
[OCCAM](https://diglib.eg.org/bitstream/handle/10.2312/cgvc20251203/cgvc20251203.pdf)
estimates coverage with no ground truth by casting randomised rays from inside
the scanned volume: **a ray that escapes without hitting anything signals
missing geometry**, because an enclosed scene should stop it. It reports 100k
points in under a second and needs no surface reconstruction. An object on a
turntable is exactly such a bounded case — arguably a cleaner one than the
indoor rooms OCCAM targets.

On separating subject from scenery, the established options are all *priors*,
not inferences — nobody claims to derive the subject from geometry alone:

- **a bounding volume**, which is what the NBV work assumes ("voxels within an
  unknown object's bounding box");
- **min-cut segmentation with a background prior** and hard foreground seeds
  ([Golovinskiy & Funkhouser](https://gfx.cs.princeton.edu/pubs/Golovinskiy_2009_MBS/paper_small.pdf));
- **multi-view foreground/background** with a Gaussian mixture plus explicit
  bottom-surface removal;
- **interactive** — the operator corrects a proposed mask.

The literature's honest position is the same as `mapper`'s: *the subject is an
input, not a measurement.*

## 5. What this repository already has and does not use

`src/scanner3d/tsdf.py` maintains exactly the three-state distinction §4
describes, and says so in its own limitations:

> "A voxel no frame ever saw keeps weight zero and is not a claim about free
> space. This module carves only where a frame actually observed something in
> front of the voxel; **unobserved volume is unobserved, not empty**."

So the fusion volume already distinguishes:

| TSDF state | meaning |
| --- | --- |
| weight > 0, distance ≈ 0 | surface |
| weight > 0, distance positive | **carved free space** — a frame saw through here |
| **weight == 0** | **never observed** |

`coverage.py` does not touch the TSDF. It bins points into cells and has only
one state. `grep tsdf src/scanner3d/coverage.py` returns nothing.

**The machinery for a correct coverage metric exists in this repository and the
coverage metric does not use it.** That is the single most useful finding here,
and it makes the leading recommendation far cheaper than it would otherwise be.

## 6. Recommendations, ranked

**R1 — Split the verdict, as `mapper` did.** Cheapest, closes the honesty gap
immediately, changes no arithmetic. Report `observed_geometry` (of what was
reconstructed, how well was it seen — the existing computation, correctly
named) separately from a subject-extent field the software declares it cannot
answer. Rename rather than supplement so every reader breaks loudly. This does
not make the number better; it stops it claiming something it never measured.

**R2 — Compute coverage over the carved volume, not over occupied cells.** Use
the TSDF's own three states. A cell adjacent to carved free space and never
observed is a *gap*; a cell in the middle of never-observed volume beyond the
subject is *not surface at all* and should leave the denominator. This is the
literature's frontier construction, and §5 says the inputs already exist. It is
motion-agnostic — nothing in it assumes what moved — which the counter-rotation
idea I considered first was not.

**R3 — Accept a subject volume as an explicit input.** The literature is
unanimous that the subject is a prior. For a turntable this is nearly free: the
subject is what is rigid about the rotation axis, and the axis is recoverable
from the poses. For handheld it has to be declared or drawn. Either way it is
*stated*, recorded in the run, and refusable — not guessed.

**R4 — Escaping-ray completeness, per OCCAM.** Once R2 exists this is a small
addition and gives a genuine "how much of the subject did I get" for bounded
objects — the question `mapper` correctly declares unanswerable for an open
course but which *is* answerable for an object of finite extent.

R1 is a day. R2 is the real work and the real payoff. R3 and R4 depend on R2.

## 7. What transfers to `mapper`

**Take from here:** the TSDF free-space/unknown distinction, and R4. `mapper`
declares `survey_extent` permanently `unknown` because an open course has no
boundary to violate — correct for a course. But `mapper` has bounded
sub-problems where the escaping-ray test applies: whether a *gate* was fully
observed, whether the volume around a landmark was swept. The scanner's case is
easier in this one respect, and the technique is portable to the parts of
`mapper`'s problem that are also bounded.

**Take from `mapper`:** the two-verdict split, the rename-don't-supplement
discipline, and its treatment of gaps — *"the gap is reported, not bridged"*,
with inferred content carrying uncertainty that grows with the gap
(`sigma = 0.25 + 0.35 × gap_m`) and a hard limit past which *"this is not a gap
to bridge, it is a section that was not surveyed"*. This scanner has no
equivalent: it fills or it does not, with no graded statement of how much it is
guessing.

**Shared, and worth noticing:** both repositories independently arrived at
ambiguity margins that refuse rather than choose — `mapper` re-solves its
assignment with each selected edge forbidden and marks the component unresolved
if an alternative ties within margin; the scanner refuses a placement when two
reference sets disagree and no cluster is `AMBIGUITY_MARGIN` tighter than the
runner-up. Same idea, same justification, arrived at twice. That is worth
factoring into shared vocabulary if the two ever share code.

## Sources

- [PB-NBV: Projection-Based Next-Best-View Planning](https://arxiv.org/pdf/2501.10663) — the four-way voxel state and the ray update rule
- [OCCAM: Occlusion-aware Completeness via Coverage Analysis with Monte Carlo sampling](https://diglib.eg.org/bitstream/handle/10.2312/cgvc20251203/cgvc20251203.pdf) — ground-truth-free coverage from escaping rays
- [Min-Cut Based Segmentation of Point Clouds](https://gfx.cs.princeton.edu/pubs/Golovinskiy_2009_MBS/paper_small.pdf) — background prior with foreground seeds
- [Next-Best-View planning for surface reconstruction](https://w3.onera.fr/copernic/sites/default/files/2020_-_iros_-_next-best-view_planning_for_surface_reconstruction_of_large-scale_3d_environments_with_multiple_uavs.pdf) — frontier extraction
- [Occlusion-Aware Visibility Coverage for Robotic Stop-and-Scan 3D Reconstruction](https://www.mdpi.com/1424-8220/26/17/5430) — visibility-based coverage (abstract only; full text was not retrievable)
- `mapper/docs/track-reconstruction.md` §6 and §8.4 — the two-verdict split and why extent is unknowable
- `mapper/docs/spatial-map-comparison.md` §3–4 — ambiguity and uncertainty-limited verdicts
