# The raw-points versus raw-TSDF-surface oracle

Status: added 2026-08-30, corrected twice the same day after review, and given
its own public CI job the same day. This exists to
answer one question with evidence instead of by eye: **why does the fused
surface look lumpier and less complete than the measured dots?** It changes no
export, live preview, GUI, capture, tracker or readiness behaviour, and it
recommends one next implementation task rather than performing it.

The contract lives in
[`tests/surface_oracle.py`](../tests/surface_oracle.py): what each arm may say
about itself, what is refused outright, and what is measured about what it
produced. The scenes live on disk as data under
[`tests/fixtures/surface_oracle/cases/`](../tests/fixtures/surface_oracle/cases),
written by [`tests/build_surface_oracle_fixtures.py`](../tests/build_surface_oracle_fixtures.py),
which is the only thing allowed to write there. One command runs the corpus:
[`tests/surface_oracle_harness.py`](../tests/surface_oracle_harness.py). The
rules are proved in Python by
[`tests/test_surface_oracle.py`](../tests/test_surface_oracle.py) and the reader
by [`tests/test_surface_oracle_harness.py`](../tests/test_surface_oracle_harness.py).
The physical companion is [`tests/surface_oracle_replay.py`](../tests/surface_oracle_replay.py).

Ordinary pytest runs both of those test modules, and the harness one runs this
corpus for real — twice, as [In CI](#in-ci) explains.
`.github/workflows/ci.yml` additionally runs the command itself as its own
named job.

## What the earlier round of this document said, and why it was wrong

An earlier revision concluded that the extractor emits an unsupported shell, on
the strength of a p95 of 0.353 and "only 86 % of vertices within tolerance" on
exact input. **That conclusion does not follow and has been withdrawn.** The
reference it was measured against was the sensor's own ideal returns, decimated
onto a lattice. A marching-cubes vertex interpolated *between* two camera rays
is far from every one of those returns and exactly on the object at the same
time, so that number was mostly a property of the reference.

Measured against the object itself, in closed form, the same arm scores a p95 of
**0.061** with **97.3 %** of vertices inside the same tolerance. Two further
things were wrong with the scope:

- the finest cell then tested was voxel 0.10, while the shipped Reconstruct
  action chooses about **0.035** on this scene. The envelope was 2.8 times
  coarser in linear resolution than the mesh an owner actually gets;
- the comparison was between two point lists — the field's zero level and the
  extractor's vertices — which by construction lie on the same field-edge zero
  crossings. That comparison cannot see a triangle spanning a gap or a shell
  sealing an unmeasured region.

A later revision fixed those three and then made the same mistake once more in
a smaller place: having separated support from truth, it read
`unsupported_closed_shells` as a defect count and recommended pruning them.
**That is also withdrawn.** Six of the seven unsupported shells this corpus
finds at the shipped resolution are analytically *correct* surface nothing
happened to measure, and the rule would have deleted them. Section 5 has the
per-arm evidence and the recommendation says what replaces it.

All four are corrected below, and every number on this page comes from the run
the corrected corpus produced.

## The one command

```bash
.venv/bin/python -m tests.surface_oracle_harness --out surface-oracle-report.json
```

- **`0`** — every arm of every case did exactly what its fixture declares, and
  the cross-case attribution reconciled.
- **`1`** — the corpus ran and something did not: an unexpected pass, a missing
  expected failure, a corruption control that was not refused, a malformed
  fixture, an attribution that does not reconcile, or an exception while scoring.
- **`2`** — the corpus could not be run at all.

Every case is attempted, so a failing artifact preserves all of them rather than
the first.

## In CI

`.github/workflows/ci.yml` runs the command above as its own job, `Surface
regression oracle`, on public GitHub-hosted `ubuntu-latest` with `contents:
read`, CPython 3.14 from the same pinned `setup-python`, the same `uv sync
--frozen --extra dev` as every other job, and every action pinned to a reviewed
full commit SHA. The step runs the exact command this page documents, once, and
nothing wraps its exit status, so **1** and **2** above both fail the job. The
whole report goes to the step log — every case, every arm, every corruption
control — so a failure can be read without downloading anything. The upload step
is guarded by `if: always()` with `if-no-files-found: error`, and the artifact
name is `surface-oracle-report`, distinct from `quality-harness-report` and
`depth-completeness-report`, so one run leaves three separately named reports
rather than a collision.

**It is not there because the ordinary suite omits this corpus.** It does not:
[`tests/test_surface_oracle_harness.py`](../tests/test_surface_oracle_harness.py)
has a module-scoped fixture that runs the committed corpus for real, and a
determinism test that runs it a second time and requires the two reports to
serialize identically, so the `Offline test suite` job already exercises it
twice. What
the dedicated job adds is that the exact documented command is run under its own
visible status, keeping its own complete report: a fused-surface regression then
reads as itself rather than as one failure inside a five-thousand-test suite,
and the report of the run that failed outlives the runner. Its timeout is 20
minutes, about six times the measured corpus cost in
[Computational cost on CPU](#computational-cost-on-cpu), which is margin for a
slower hosted runner while still catching an oracle that hangs.

**Hosted and CPU-only on purpose, and a GPU is deliberately not the correctness
oracle here.** The six declared grid cells run on the pure-Python reference
backend; the shipped-policy appliance arm runs on the numpy CPU backend, because
the resolution that policy derives needs about two million voxels — roughly
thirty times `scanner3d.tsdf.PYTHON_BACKEND_MAX_VOXELS` — and the artifact
records that split among its own limitations. Both are CPU implementations
scored against committed synthetic truth, and no accelerated implementation is
selected or required anywhere in this corpus, so an accelerator would add a
different execution path without improving the correctness evidence — while
making the run depend on a machine no reviewer of this repository can inspect.

It is the **third** named CPU correctness-oracle job, beside `Reconstruction
quality harness` ([docs/quality-harness.md](quality-harness.md)) and
`Depth-completeness benchmark`
([docs/depth-completeness-benchmark.md](depth-completeness-benchmark.md)). The
three are separate jobs because they measure different evidence and keep
different artifacts. [`tests/test_ci_workflow.py`](../tests/test_ci_workflow.py)
parses the workflow offline and asserts all of the above — the runner, the
read-only permissions, the pinned action identities, the frozen sync, the exact
command run exactly once, that no step or job swallows a failure, that each job
uploads the file its own command wrote under its own name, that this job carries
its justified 20 minutes and the ordinary suite its 60, and that nothing
anywhere reaches for a self-hosted runner, an accelerator, hardware or a
deployment path. It never dispatches Actions and never uses the network.

## The method

One exact-truth multi-view scene is rendered once by `scanner3d.simulate`. A
fixture declares one *perturbation* of it — and only one, except in the
deliberately combined case — and **the same raw frames and the same poses** go
to three kinds of arm:

- **`raw_points`** — the unchanged measured point set, deprojected through the
  production path and placed by the same poses the fusion is given;
- **`fused:<cell>`** — the unchanged `scanner3d.tsdf` field integrated from
  those same frames and poses at one predeclared parameter cell, meshed by the
  unchanged `scanner3d.tsdf_surface.extract_surface`;
- **`fused_at_appliance_resolution`** — the same two modules at the resolution
  the **shipped** policy chooses for this scene.

Every metric is computed by the oracle, from the arm's output and the scene's
exact truth, and never taken from the arm.

### Two references, and they are not the same thing

| | what it is | what it is not |
| --- | --- | --- |
| **`analytic`** | the exact closed-form distance to the boundary of the union of the scene's declared shapes, plus coverage of a dense exact resampling of the same surface | it does not know where any camera ray landed |
| **`support`** | proximity to the ideal returns the five views produced, decimated onto the fixture's lattice | **it is not accuracy.** A point on a face no view could see is correct geometry and unsupported measurement at once |

`analytic_distance` is closed form over disjoint shapes, and
`check_shapes_are_separable` refuses a scene whose shapes touch rather than
quietly reporting a bound. The analytic *coverage* reference is built by
intersecting sub-pixel camera rays with the shapes in closed form at the
supersample factor the fixture owns — 51 200 exact points here, five times the
sensor's own sampling — never by reading a rendered frame back.

### The surface is measured as a surface

Vertices alone are not enough, and the reason is a property of marching cubes:
an extracted vertex sits on the same field-edge zero crossing the pre-extraction
samples report, so comparing those two point lists can only find small
disagreements. Two measurements fix that:

- **`surface_samples`** scores a fixed, strictly interior barycentric lattice
  inside *every* emitted triangle — order 3, which is **nine equal-area samples
  per face**: one centroid per sub-triangle of a regular subdivision, upright
  and inverted — against the same analytic truth and the same protected volume.
  Equal area is not a detail. It is what lets the fraction of a face's samples
  that land in the protected gap be read as the fraction of its *area* that
  does, which is the inference `protected_area` makes. It publishes
  `bridging_faces` and, separately,
  `bridging_faces_with_every_vertex_outside`: triangles that cross the declared
  empty gap while all three of their corners stay out of it, which no vertex
  count can see. The order and a sample ceiling are owned by the fixture, so the
  work is bounded before any result exists and nothing is thinned afterwards.
- **`topology`** counts what the faces *are*: `scanner3d.mesh.edge_census` for
  manifoldness and closure, the shells the face graph breaks into, the Euler
  characteristic, and — the number the complaint is really about —
  `unsupported_closed_shells`, watertight shells with no ideal return within the
  coverage tolerance of any of their vertices. Each of those is then scored
  against the analytic surface as well, at its own corners and inside its own
  triangles, and published as two disjoint groups:
  `unsupported_inaccurate_closed_shells`, which are off the object and are
  therefore false surface, and `unsupported_accurate_closed_shells`, which are
  on it. **Unsupported is not false.** A sealed shell standing where no view
  had a ray can be exactly on the object, and the second group is published so
  that the cost of any support-only rule is visible rather than assumed.

Both are inapplicable to the raw-points arm, which is a cloud and has no faces.
That is recorded as inapplicable, never as passing.

### The object has four features on purpose

One asymmetric object seen from five views carries **a plane** (the backing
surface), **a curved surface** (a sphere), **a hard depth step** (a box standing
four units in front of the plane) and **a thin feature** (a pillar 0.2 units
across, about 2.8 pixels wide at its own depth). They are together because every
parameter that smooths one of them damages another: **no cell of the grid can
pass by erasing the geometry it could not represent.**

### The seven cases

| case | what it varies | control |
| --- | --- | --- |
| `exact-control` | nothing | — |
| `depth-noise` | bounded zero-mean noise, ±20 codes (0.1 units) | `exact-control` |
| `depth-outliers` | ±60 codes on every 37th return, alternating sign | `exact-control` |
| `no-return-bands` | 2 of every 7 columns drop, walking 3 columns per view | `exact-control` |
| `pose-translation` | 0.05-unit lateral bias plus 0.05-unit per-view jitter | `exact-control` |
| `pose-rotation` | 0.3-degree bias plus 0.3-degree per-view jitter | `exact-control` |
| `combined-realistic` | depth noise, no-return and both pose errors together | `exact-control` |

A pose perturbation moves **only the pose the reconstruction is given**. The
frames stay byte-identical to the truth, which is exactly what a tracking error
is. The label is load-bearing: `check_perturbation_label` binds it to the fields
that are actually non-zero, so a case cannot be filed under "pose error" while
quietly carrying depth noise.

### The grid is owned by the corpus, and one arm is bound to the appliance

Six declared cells, in the manifest before any result existed, varying one thing
at a time from the first: `v15-t3-w1-uniform` (voxel 0.15, truncation 3 voxels,
minimum weight 1), then minimum weight 2, truncation 2, `ramped_behind`
weighting, voxel 0.25 and voxel 0.10 with truncation 4. **Every case runs every
cell.** A case that did not is refused before a cell is chosen. The work
ceiling — 500 000 voxel-view units — is the manifest's, and an arm that spends
past it is refused.

Beside them, one arm that is not a cell at all.
`fused_at_appliance_resolution` runs
`scanner3d.reconstruction_export.choose_tsdf_resolution` on the case's own
measured points at the desktop Reconstruct action's voxel budget (`128 ** 3`),
with the truncation ratio, minimum weight and weight policy
`scanner3d.reconstruction_export` exports today —
`check_appliance_policy` refuses a manifest whose declared constants have
drifted from the imported ones, and the budget is bound the same way: it is
compared with `scanner3d.gui.RECONSTRUCT_VOXEL_BUDGET` itself, because the
budget is the number that decides how fine this arm is and a fixture free to
write any value there could name a resolution no owner ever gets. Its voxel
size is **derived and published, never
declared**: 0.0352 to 0.0370 across the seven cases, against a finest declared
cell of 0.10. That is 1.73 to 2.01 million voxels rather than 86 632, and it is
the only arm that describes the mesh an owner actually gets. It runs on the
numpy backend because it is thirty times past the pure-Python voxel ceiling,
which is stated in the artifact's own limitations; every declared cell runs on
the pure-Python reference so two machines agree.

The recommended cell is chosen by a rule stated in code *and* repeated verbatim
in the manifest, which the loader checks: highest **analytic** surface coverage,
ties broken on the lower analytic p95, then lower work, then lexicographic cell
id. The appliance arm is reported beside the winner and is never eligible to be
it, because its resolution is not a choice the grid offers.

## What the corpus measured

`decision: pass`, 7 of 7 cases, 105 of 105 candidates. Every number below is
from that artifact.

### 1. The extractor is not the cause, and the ablation says so

Every fused arm publishes an `ablation`: the field's own zero level from
`TsdfVolume.zero_level_samples`, with no marching cubes anywhere near it, and
the extractor's output, scored against the *same* analytic truth by the same
code.

- `extraction_p95_delta` lies between **−0.0265 and +0.0091** on every fused arm
  of every case — a fraction of one voxel at every cell that produced it;
- `surface_points_without_field_support_fraction` is **0.0** on all of them.

The extractor reproduces the field it is handed. What remains is fusion's:
discretisation, pose and depth. That is the attribution the previous revision
asserted in the other direction without evidence.

### 2. At the resolution an owner actually gets, the surface is accurate

`exact-control` — no noise, no outliers, no missing returns, exactly correct
poses:

| measure | `raw_points` | `fused:v10-t4-w1-uniform` | `fused_at_appliance_resolution` |
| --- | ---: | ---: | ---: |
| voxel size | — | 0.100 | 0.0352 |
| analytic accuracy p95 | 0.0023 | 0.0613 | **0.0193** |
| vertices within tolerance of the object | 1.000 | 0.973 | **1.000** |
| analytic surface coverage | 0.9997 | 0.9981 | 0.9963 |
| triangle-interior p95 | — | 0.0652 | **0.0159** |
| triangle-interior samples within tolerance | — | 0.978 | **1.000** |
| bridging triangles | — | 0 | 0 |
| unsupported closed shells | — | 0 | 0 |
| of those, false (off the object) | — | 0 | 0 |
| measurement support (observed→returns coverage) | 1.000 | 0.856 | 0.9997 |
| point count | 9898 | 3434 | 21509 |

The 0.856 in the middle column is the number the earlier revision called an
extraction defect. Read the appliance column instead: at the resolution the
product uses, the mesh is inside tolerance of the object at every vertex *and*
at every interior sample of every triangle, and 99.97 % of it is near something
a view measured.

### 3. Fusion beats the dots exactly where averaging should help

Analytic accuracy p95, dots against the appliance arm:

| case | `raw_points` | appliance arm | better |
| --- | ---: | ---: | --- |
| `exact-control` | 0.0023 | 0.0193 | dots |
| `no-return-bands` | 0.0023 | 0.0180 | dots |
| `depth-outliers` | 0.0024 | 0.0917 | dots |
| `pose-rotation` | 0.0454 | 0.0636 | dots |
| `pose-translation` | 0.0637 | 0.0718 | dots |
| `depth-noise` | 0.0950 | **0.0828** | **fusion** |
| `combined-realistic` | 0.1238 | **0.1077** | **fusion** |

Bounded zero-mean noise is what a running weighted mean is for, and on the two
cases that carry it the fused surface is more accurate than the dots that
produced it. Under outliers it is not: an outlier is not cancelled by averaging,
it drags the zero crossing of every voxel it touches. On the plane, under the
combined degradation, the surface is also smoother than the dots — p95 residual
0.0919 against 0.1099 — while on exact input the dots are smoother (0.0025
against 0.0084).

Across the sixteen attributed metrics a point cloud and a mesh both publish —
the corpus declares twenty-nine in all, the rest of which only a mesh has —
fusion is better on **1 to 4 of 16** at
the recommended grid cell and on **2 to 7 of 16** at the appliance resolution.
The appliance resolution is better than the declared envelope on every case.

### 4. Bridging is a coarse-grid failure, and it disappears at the appliance resolution

`edge.bridged_points` (vertices in the protected empty gap) and
`surface_samples.bridging_faces` (triangles whose interiors enter it):

| case | `fused:v10-t4-w1-uniform` | appliance arm |
| --- | --- | --- |
| `pose-translation` | 54 vertices, 76 triangles | **0, 0** |
| `pose-rotation` | 54 vertices, 76 triangles | **0, 0** |
| `combined-realistic` | 12 vertices, 21 triangles | **0, 0** |
| every other case | 0, 0 | 0, 0 |

Bridging is uniquely attributable to pose error, and it is removed entirely by
the resolution the product already uses. On this corpus the triangle count and
the vertex count agree about *where* bridging happens — every bridging triangle
here also has a corner in the gap, so
`bridging_faces_with_every_vertex_outside` is 0 everywhere. That the two agree
is a result about this corpus, not a reason to drop the triangle measurement:
`tests/test_surface_oracle.py` builds a triangle spanning the gap with all three
corners outside it and shows the vertex count reporting nothing while the
interior sampling reports one bridging face, and the
`fused_with_bridging_triangle` control puts the same forgery in front of the
reader with a byte-identical vertex list.

### 5. Sealed shells nothing measured — and why only one of them is a defect

`topology.unsupported_closed_shells` counts watertight shells with no ideal
return within the coverage tolerance of any vertex. Every arm in the corpus that
has any, at every resolution:

| case | arm | unsupported shells | **false** (off the object) | **correct** (on it) | faces at stake | worst analytic distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `depth-outliers` | `fused:v10-t4-w1-uniform` | 1 | **1** | 0 | 8 of 7 194 | 0.3042 |
| `depth-outliers` | appliance | 1 | **1** | 0 | 40 of 38 814 | 0.3030 |
| `combined-realistic` | `fused:v15-t3-w1-uniform` | 1 | **1** | 0 | 16 of 3 310 | 0.3452 |
| `combined-realistic` | `fused:v25-t3-w1-uniform` | 1 | **1** | 0 | 8 of 1 142 | 0.5303 |
| `combined-realistic` | appliance | 6 | **0** | **6** | 100 of 70 063 | 0.1247 |
| every other arm | — | 0 | 0 | 0 | — | — |

**An earlier revision of this page called every one of those shells a defect and
recommended pruning them. That was wrong, and the last row is why.** The six
shells the combined case seals at the shipped resolution are analytically
*correct*: every vertex and every sampled triangle interior is within 0.1247 of
the object, inside the same 0.15 tolerance every other accuracy number on this
page uses. Nothing measured them because nothing could — they stand where no
view had a ray — and a rule that deleted unsupported shells would remove 100
faces of correct surface from the case that most resembles a real capture, in
exchange for removing no false surface at all there.

The false shells are a different thing and they are a real defect: 0.30 to 0.53
from the object, three to four times the tolerance, and produced by outliers
(`depth-outliers`) or by a coarse grid under combined degradation. At the
resolution an owner actually gets, exactly **one** false shell exists in the
whole corpus, and it is under outliers. Support alone cannot separate the two
groups — that is the measurement, not an opinion — so this page gates
`no_false_closed_shells` on the analytic half and publishes the other half
beside it.

All of these arms pass every point-based lump gate: `no_isolated_lumps` and
`body_count` are satisfied, the distance summaries are unremarkable, and the
shells' vertices are ordinary zero crossings of the field. Nothing in the
previous revision's vocabulary could have found them at all, and nothing in the
revision after that could have told them apart.

Note what `combined-realistic` is: depth noise, structured no-return and both
pose errors, and **no outliers**. None of those three perturbations produces a
single unsupported shell on its own; together they produce six. That
interaction is real and is what the combined case exists to expose. What it
produces here is correct geometry nobody measured, not invented solid.

### 6. One dimension the oracle cannot yet attribute

`dimension_error` — the object's three named extents against exact truth — fails
on the appliance arm in five of seven cases, and it is always the *z* extent.
On `exact-control` the appliance arm's x and y extents are better than any
declared cell's (0.077 and 0.079 against v10's 0.150 and 0.195) while its z
extent is 0.41 short. Across the grid the sign is not monotone in either voxel
size or truncation: v15 cells run 0.45 to 0.68 short, v10 and v25 run about 0.10
long, the appliance arm runs 0.41 short.

**This oracle does not attribute that.** It is recorded as an open measurement
rather than explained, and nothing below recommends acting on it.

### 7. A boundary effect that is not invented geometry

`points_inside_volume` fails on three appliance arms — 148 of 21 509 vertices on
`exact-control`, 32 of 40 519 on `depth-noise`, 161 of 24 338 on
`pose-translation`. The cause is arithmetic:
`scanner3d.tsdf.TsdfConfig.shape` rounds the voxel count **up**, so on an axis
whose extent is not a whole multiple of the voxel size the outermost voxel
centre sits just past the declared box — at most half a voxel, which is 0.018
units here and well inside the fixture's own 0.05 bounds margin. The backing
plane reaches that far in y. It is a property of the measuring box, not of the
mesh, and the gate limit was **not** moved to hide it.

## The tested parameter envelope

The recommended cell is `v10-t4-w1-uniform` — the finest declared voxel — in six
of seven cases, and `v15-t3-w1-uniform` on `pose-translation`. Coarsening never
helped anything: `v25-t3-w1-uniform` is the worst cell in the corpus, and it is
the only cell that fails `analytic_accuracy_p95` and
`triangle_interior_accuracy_p95` in every single case.

- **Minimum weight 2** (`v15-t3-w2-uniform`) trades coverage for support and
  costs `support_coverage` on the no-return and combined cases. Not a fix.
- **Truncation 2 voxels** (`v15-t2-w1-uniform`) produces isolated lumps on the
  *exact control* — a shorter band commits to less, and what is left at the
  pillar breaks off. Not a fix.
- **`ramped_behind` weighting** (`v15-t3-w1-ramped`) improves the object's x and
  y extents and makes the z extent worse (0.675 against 0.451 at uniform), and
  it loses 92 % of the pillar on the combined case. Not a fix.
- **Voxel 0.10 with truncation 4** is the best declared cell everywhere and is
  still 2.8 times coarser than what the product uses.

**No cell of this grid is recommended as a change**, and none is needed: the
shipped policy already chooses finer than every one of them, and the appliance
arm is better than the best declared cell on every case.

## Computational cost on CPU

Every declared cell runs on the pure-Python backend so two machines agree; the
appliance arm runs on the numpy CPU backend, and both are CPU implementations.
One full run of the corpus — 7 cases, 105 arms — takes about three minutes and
about 525 MB of resident memory on the development host, measured at 3:10.9 for
the run this page reports. That measurement is what the CI job's 20-minute bound
is a multiple of. The deterministic cost measure is `work_units`, voxels
times integrated views:

| arm | cells | work units | ceiling |
| --- | ---: | ---: | ---: |
| `fused:v10-t4-w1-uniform` | 86 632 | 433 160 | 500 000 |
| `fused_at_appliance_resolution` | 1 725 276 – 2 009 840 | 8 626 380 – 10 049 200 | 14 000 000 |

Both ceilings are the manifest's and were stated before any result existed. Wall
clock is never a gate and never appears in the artifact.

## The corruption controls

Eight forgeries run beside the honest arms, each of which must earn one named
refusal, and each of which is exactly the thing somebody would actually do:

| control | refusal | what it does |
| --- | --- | --- |
| `fused_from_truth_frames` | `raw_frame_edited` | fuses the exact truth frames and presents the surface as the measured one. Its geometry really is better than any honest cell's — the impossible improvement. |
| `fused_dropping_a_view` | `views_omitted` | leaves out the view that disagrees. |
| `fused_with_true_poses` | `pose_substituted` | uses the poses the scene kept back, which no reconstruction has. |
| `fused_with_edited_surface` | `surface_not_derived` | deletes the vertices standing in free space after extraction. |
| `fused_with_bridging_triangle` | `surface_not_derived` | keeps every honest vertex exactly where the extractor put it and adds **one triangle** spanning the protected gap. The point list is byte-identical to the honest arm's, so every vertex-based measurement agrees with it to the last bit. |
| `raw_points_dropping_hard_samples` | `sample_count_mismatch` | the raw arm's own cheat: drop the measurements at the depth step and report a cleaner cloud over a smaller denominator. |
| `fused_overclaiming_work` | `claim_not_supported` | halves its declared voxel-view spend. |
| `fused_claiming_the_unmeasurable` | `claim_unknown` | publishes a `surface_accuracy_target` of its own. |

Two of them are only *detectable* on some cases: fusing the truth frames is
byte-identical to fusing the measured ones when the case has no depth
perturbation, and substituting the true poses is identical when the case has no
pose perturbation. They are therefore declared only where the perturbation makes
the difference visible, and
`tests.surface_oracle_harness.CONTROL_REFUSALS` refuses a corpus in which any of
them stopped being expected anywhere — which is what a quietly deleted bad
control looks like from the outside.

The reader is attacked separately. A swapped baseline (the control's rendered
truth must be the perturbed case's truth, view for view), a control that is
itself perturbed, a cherry-picked grid cell, an artifact with no appliance arm, a
manifest with no appliance policy or one whose constants have drifted from the
product — including a voxel budget that is not
`scanner3d.gui.RECONSTRUCT_VOXEL_BUDGET` — a triangle sampling that skipped part
of its own mesh, one whose per-face count is not the `n**2` the equal-area
lattice produces, a sampling that ran past its
own ceiling, more hidden bridging faces than bridging faces, a topology closure
flag that contradicts its own edge counts, more unsupported shells than closed
shells, a shell split that loses or double-counts an unsupported shell, an
unsupported shell filed as false while its own numbers put it on the object, a
shell filed as correct while its own numbers put it off the object, a split
described as having been made against the support reference, an analytic
reference described as something else, a coverage over its
own reference, an ablation delta that is not its own subtraction, an edited
attribution delta, a flipped "degraded" flag, a case counted twice, a decision
that is not the arithmetic of its own statuses, and a manifest that does not
match the bytes on disk are each refused, and
`tests/test_surface_oracle_harness.py` proves each one fails for its own reason.

## The physical replay, and what it found

```bash
.venv/bin/python -m tests.surface_oracle_replay
.venv/bin/python -m tests.surface_oracle_replay --capture /home/ben/scans/scan/20260828-110641 --limit 12
.venv/bin/python -m tests.surface_oracle_replay \
  --capture /home/ben/scans/scan/20260827-152842 --limit 12 --backend numpy
```

Strictly read-only, and it proves it: size, mtime and SHA-256 of every file in
the capture are taken before and after, and the report carries both. There is no
truth model for a physical capture, so **no analytic accuracy, truth coverage,
dimension error, edge coverage, thin-feature survival, triangle-interior
accuracy or shell-support verdict is measured or claimed there** — only return
coverage, pose acceptance, roughness proxies over a region the caller names,
components, and parameter sensitivity. The declared appliance policy is
published as context; a run that stops before fusion derives no resolution at
all.

Every coordinate the replay measures, publishes or hands to a field is in the
**model frame** — the frame the registration's poses are in, which is the frame
`TsdfVolume.integrate` documents for the poses it takes and the frame
`export_reconstruction` measures its bounds in. The presentation half turn
`UPRIGHT_FROM_OPTICAL` is applied to nothing here, because nothing here is
reported in the upright display/export frame. It is worth saying because the
replay got this wrong: it bounded the field from `model_points(clouds, poses)`
and then fused at `upright_pose(poses)`, so every voxel projected half a turn
away from the raw evidence that should have filled it and every field came back
empty. A second defect hid the first — the fusion window was a synthetic
scene's `0.05`/`4.0` rather than the appliance's own `TSDF_MIN_DEPTH` and
`TSDF_MAX_DEPTH`, and a real capture's returns lie entirely outside it, so no
observation of one survived projection whatever the poses were. Both are fixed,
and `tests/test_surface_oracle.py` now holds a fused control that fails on
either.

### Three captures, and what each one is evidence of

Run on 2026-08-30. These are the captures available to this repository — a
fourth may exist tomorrow, and this list is not a claim that physical evidence
is exhausted:

- `tests/fixtures/turntable` (24 frames, manifest-bound, every frame's digest
  verified before use): **stopped at registration.** Frame 3 could not be
  placed — the adjacent placement moves the surface 21.994 and the keyframe
  placement 24.158, against a 15.600 gate. Read-only verified across 25 files:
  nothing changed, appeared or went missing.
- `/home/ben/scans/scan/20260828-110641` (12 frames of 166 used, bound by frame
  digests; the capture carries no manifest): **stopped at registration.** Frame
  1 could not be placed; only 2 of 492 source points found a normal-bearing
  partner inside the correspondence gate. Read-only verified across 166 files.
- `/home/ben/scans/scan/20260827-152842` (12 frames of 116 used, bound by frame
  digests; no manifest): **reaches fusion, and is the first one that does.**
  Registration placed all 12 frames with no revisit found and no pose-graph
  optimisation; every readiness gate passes. 46.345475260416674% of the 256000
  samples per frame carried a return **on average** — the twelve-frame mean,
  which is also the aggregate fraction because every frame has the same 256000
  samples. It is not a property of each frame: the individual fractions range
  from 39.56015625% to 48.48203125%. The chooser derived voxel 0.545 over a
  measured extent of 128.0 x 117.5 x 139.1, one coarsening step, a 243x224x264
  field of 14,370,048 voxels. All 12 frames changed every allocated field:
  2,725,894 observed voxels and a 73,170-vertex, 130,346-face surface at the
  base cell `v15-t3-w1-uniform`, and a non-empty surface at all five allocated
  cells. The finest cell `v10-t4-w1-uniform` is refused for wanting 48,442,800
  voxels against the module's 16,777,216 ceiling, which is the ceiling behaving
  correctly and not a result. Read-only verified across all 116 files: nothing
  changed, appeared or went missing.

The two registration refusals are results and are preserved as ones: the
harness does not register a prefix, skip the frame or manufacture a pose in
order to produce numbers. Neither was re-run, re-cut or re-gated to make it
reach fusion, and `20260827-152842` was not selected after seeing them fail: it
is a capture the earlier notes did not know about.

**What `20260827-152842` proves.** That the whole path runs on real returns:
that a physical capture of this scanner can be prepared, registered, pass every
shipped readiness gate, fill a field through the appliance's own depth window at
the poses the registration fitted, and yield a non-empty surface at five of the
six declared parameter cells. It also fixes the shape of the response — how the
observed-voxel count, the vertex and face counts and the component counts move
with voxel size, truncation, minimum weight and weight policy — on measured
evidence rather than on rendered evidence.

**What it does not prove, and cannot.** There is no truth model for it, so it
carries **no independent truth of any kind**: no analytic accuracy, no
dimensional accuracy — no number in that report has a physical unit at all, and
the depth unit's physical meaning is still unestablished — no edge accuracy, no
thin-feature survival, no shell-support verdict, and no statement whatever about
whether depth completion or weighted input would improve any of it, since
neither is routed into the replay or into the product. The poses the gates judge
and the poses the fusion is given were both fitted by that same run, so a
registration error the fitter was confident about is folded into the surface
rather than measured. A non-empty surface is not a correct surface: what is
demonstrated is that the arithmetic completes on real returns, not that the
geometry it produces is right. **Every causal claim in this document still rests
on synthetic exact truth alone**, and this capture is the reason to treat that
truth as necessary evidence rather than sufficient — not a replacement for it.

## What this corpus cannot say

- It is synthetic. `scanner3d.simulate` is not a sensor model — no speckle
  physics, multipath, subsurface scattering or baseline shadowing. A cause
  isolated here is a cause *of the arithmetic*, demonstrated on exact truth.
- The perturbation magnitudes are constructed, not fitted to a capture. They are
  not claims about what the MINI does.
- No number is a physical length. Depth is in synthetic units of raw code times
  depth scale, and no dimensional-accuracy claim is made anywhere.
- The two references are not interchangeable, and neither is called the other's
  name. `analytic` is the object; `support` is where the sensor sampled.
- The triangle interiors are sampled on a fixed equal-area lattice at order 3 —
  nine samples per face. A defect entirely between two adjacent samples of one
  triangle is not seen; the order is published so a reader can judge it against
  the voxel size beside it.
- The topology evidence counts what the faces are, not whether the object
  *should* have been closed. A closed shell is not itself a defect, and neither
  is an unsupported one. Only a shell that is unsupported **and** off the object
  is gated, and the accurate/inaccurate split is made at the coverage tolerance
  over the shell's vertices and its sampled triangle interiors: a shell whose
  error hid entirely between two adjacent interior samples and away from every
  corner would be read as accurate.
- The appliance arm's last bits are machine-dependent because it needs the numpy
  backend. Every declared cell is pure Python and is not.
- The z-extent error in section 6 is measured and **not** attributed.
- Task 032's negative result is preserved by construction: **no completion of
  any kind is offered to a volume here.** The only input a fused arm may
  integrate is the case's own raw measured frames.

## The recommendation

**Keep delivering the dots as the primary representation while the next cause is
addressed.** This revision recommends **no product change at all**: both fixes
earlier revisions proposed are withdrawn below, and what replaces them is one
bounded offline experiment.

**Withdrawn: "restrict surface extraction to the observed region."** It was
justified by a measurement against the wrong reference. At the shipped
resolution the extracted surface is inside tolerance of the object at every
vertex and every triangle interior on the exact control, and the ablation shows
extraction reproducing its field to within a fraction of a voxel everywhere. A
change that pruned everything far from a measured return would delete correct
geometry to fix a number that was never an error.

**Also withdrawn: "prune connected shells that no view supports."** A previous
revision of this page recommended it on the strength of
`unsupported_closed_shells` alone. Measured against the object as well, that
number does not support it. Of the six unsupported shells the combined case
seals at the shipped resolution, **six are analytically correct** and none is
false; the rule would delete 100 faces of correct surface there and remove
nothing. The one place it would remove real false surface — the single shell
under outliers, 40 faces — is also the one place a support-only rule and a
truth-aware rule happen to agree, and the corpus cannot tell the product which
of the two it is looking at, because the product has no truth model. Support is
observable at capture time; analytic accuracy is not.

**The next task is therefore an experiment, not a change.** What this corpus
establishes is that `unsupported_closed_shells` splits into two groups that
nothing this corpus currently measures *outside* its truth model separates, and
that at the shipped resolution the case most like a real capture has six
correct such shells and none that is false. Before any pruning ships, the open
question is whether some
*observable* shell property — shell size, the number of distinct views whose
frusta contain it, distance to the observed frontier, the fused weight carried
by its voxels — separates the analytically false shells from the analytically
correct ones on this same corpus. That experiment is offline, needs no product
change, and would either produce a rule with a measured cost and benefit or
prove there is none. Until then the answer for the mesh is the sentence above:
**keep delivering the dots.**

**That experiment has now been run, and it produced the negative result.** See
[docs/shell-discriminant.md](shell-discriminant.md). Its first pass inspected
its own holdout during development and then widened the threshold grid, so that
pass is development evidence and all nineteen of its members are training data;
the result below is the sealed run, whose eight holdout members, seeds,
magnitudes and 195 frozen candidate bodies were declared before any of them was
executed.

27 members at two resolutions produced 776 closed shells, of which 140 are
unsupported: 117 false (2,224 faces) and 23 correct (476 faces). At the
**shipped** appliance resolution — the only one training eligibility and the
holdout gates may read — that is 32 training and 41 sealed holdout records, both
carrying both classes. None of the 195 predeclared candidates removed at least
half the training split's false shell faces there without removing correct ones,
so no rule was selected, no holdout gate was evaluated and no counterfactual
ran. The whole search cost one bounded pass of 27,300 rule decisions against a
40,000 ceiling.

The truth-fed upper bound the planner predeclared on a sealed member could not
be measured — that member produced no closed shell at the appliance resolution —
and is published as unmeasured rather than moved. As development evidence, a
perfect truth-fed prune on the training member `outliers-140-p17` removes 100
false faces, leaves 92 correct unsupported faces before and 92 after, and moves
analytic vertex accuracy by 0.2207 percentage points across 0.2317 % of the
triangles. Pruning is rejected; nothing in this corpus changes.

No pruning is recommended, and no gate here should be read as one: the corpus
gates `no_false_closed_shells`, which is a statement about *this* corpus's
truth model and not a rule the product could evaluate.

Two things this corpus deliberately does **not** recommend acting on: the
z-extent error, because it is unattributed; and any cell of the parameter grid,
because the shipped policy already chooses finer than all of them and beats all
of them here.

Neither change is made here. This corpus measures; the planner decides.
