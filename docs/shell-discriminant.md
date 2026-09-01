# Can an appliance tell a false sealed shell from a correct one?

No. Not on the sealed holdout this round declared before it ran, and not well
enough to prune anything.

The accepted surface oracle ([docs/surface-oracle.md](surface-oracle.md))
established the fact this experiment starts from. A fused surface contains
**unsupported closed shells** — sealed, watertight blobs standing where none of
the scene's views measured anything — and they are not one thing. Some are
false surface: nothing measured near them and they are not on the object, so
removing them is pure gain. Some are correct surface: geometry that really is
on the object and that nothing happened to measure, so removing them is pure
loss. The accepted artifact publishes those two groups separately and refuses
to collapse them, which is why "delete what nothing supports" has never been
allowed near the product.

This corpus asks the next question, and it is empirical rather than an opinion:

> Using only evidence the appliance actually has, can a bounded rule separate
> the false shells from the correct ones — removing a material amount of false
> surface while removing none of the correct surface?

Run it with:

```bash
.venv/bin/python -m tests.shell_discriminant_harness --out shell-discriminant-report.json
.venv/bin/python -m tests.build_shell_discriminant_fixtures   # regenerate the population, then read the diff
.venv/bin/python -m tests.shell_discriminant_replay           # the physical companion, strictly read-only
```

The offline tests around it — `tests/test_shell_discriminant.py`,
`tests/test_shell_discriminant_harness.py` and
`tests/test_shell_discriminant_replay.py`, including one run of the full
producer over the committed corpus — run in the **ordinary pytest job on the
existing public GitHub-hosted workflow**. No self-hosted, LAN, heavy-machine or
GPU routing was added, and `.github/workflows/ci.yml` is unchanged.

## Round one was development evidence, and is labelled as such

The first pass of this experiment inspected every member of its own holdout
during development and then widened the candidate threshold grid. Its
generalization claim was therefore not one, and this document does not repeat
it. **All nineteen of round one's members are training data now.** Anything in
this file attributed to `outliers-140-p17`, `combined-strong`, `noise-24` or any
other non-`sealed-` member is *development/pilot evidence*: useful for showing
what the machinery measures, never a held-out result.

## The sealed holdout, declared before it ran

Eight members were declared by the planner — id, family, every perturbation
magnitude, and the seed `shell-discriminant:<member-id>` — and written into
`tests/build_shell_discriminant_fixtures.py` **before any builder, probe, scene
construction, fusion or harness run touched a sealed id**:

| member | family | magnitudes |
| --- | --- | --- |
| `sealed-outlier-125-p41` | depth-outliers | outlier codes 125, period 41 |
| `sealed-outlier-185-p43` | depth-outliers | outlier codes 185, period 43 |
| `sealed-outlier-155-p47` | depth-outliers | outlier codes 155, period 47 |
| `sealed-outlier-105-p53` | depth-outliers | outlier codes 105, period 53 |
| `sealed-combined-a` | combined | noise 19; bands 2/7 shift 4; translation bias [0.052, 0, −0.006], jitter 0.047; rotation bias 0.31°, jitter 0.27° |
| `sealed-combined-b` | combined | noise 23; bands 3/19 shift 4; translation bias [0.064, 0, 0.012], jitter 0.056; rotation bias 0.38°, jitter 0.34° |
| `sealed-combined-c` | combined | noise 17; bands 4/19 shift 5; translation bias [0.046, 0, −0.013], jitter 0.041; rotation bias 0.26°, jitter 0.33° |
| `sealed-combined-outlier-d` | combined | noise 21; outliers 135/43; bands 2/11 shift 4; translation bias [0.057, 0, 0.008], jitter 0.049; rotation bias 0.34°, jitter 0.28° |

The manifest records that list under `sealed_holdout`, the builder refuses to
write a population whose holdout half is not exactly it, and `load_corpus`
refuses a corpus whose holdout membership has gained, lost or renamed one.

The 195 candidate bodies were frozen at the same moment. Their **candidate-only**
digest — SHA-256 over the candidate schema version, the threshold grid, the
declared pairs, every rule body and the sorted observable names, and
deliberately *not* over the selection protocol or the gate names — is

```
4fe6627ea7618b372cf833eae805547420fa91c412808e1e12701b1191178ca2
```

`check_candidates_are_frozen()` refuses to run at all if it moves. The broader
protocol digest is separate and *did* move this round, because the selection
protocol and two gate names were legitimately rewritten; that is exactly the
distinction the two digests exist to keep.

`MATERIAL_FRACTION` is 0.5 and `WORK_CEILING` is 40,000 producer rule/shell
decisions, both unchanged. The selection resolution is predeclared as
`appliance`, and the truth-fed upper bound was predeclared as
`sealed-outlier-185-p43` at that resolution.

## The wall between the two namespaces

Round one put the truth wall at the selector and then handed the derivation a
whole `Scene`, the ideal-return support and the analytic scoring in one call, so
"the observables are truth-free" was a property of how a function happened to be
written. This round the wall is **what the derivation can reach**.

`ObservableInputs` is an immutable value carrying only: the ordered raw frames,
the fused poses actually used, the intrinsics, the scoring window, the depth
scale and bounds, the extracted vertices and faces, the cell's voxel size,
truncation, minimum weight and weight policy, the volume box, and the kept
surface-generating field. It has nowhere to put a `Scene`, the declared shapes,
the ideal support, a truth frame, a true pose, a `SurfaceScoring`, a
perturbation family, a split or an analytic label, and `check_inputs` refuses a
field that merely *reaches* an object exposing those.

`derive_observables(inputs)` is a pure function of that value and returns **every
closed shell** of the mesh, observable only. The evaluator then, separately,
uses the scene's ideal returns to filter that list to the unsupported ones and
`tests.surface_oracle.analytic_distance` to attach a `ShellTruth` to each. The
member, family, split and resolution live on the evaluator's `ShellRecord` and
on report metadata; the observable record carries none of them. Its identity is
`<first 16 hex of the reconstruction-input digest>|<shell index>`.

A test patches `Scene.shapes`, `Scene.truth_codes`, `ViewEvidence.truth`,
`ViewEvidence.true_pose`, `truth_points`, `truth_surface`, `analytic_distance`,
`score_surface` and `_covered_mask` so every one of them raises, proves the
patches bite, and then shows `derive_observables` still produces byte-identical
output.

`select` is still the only way a rule is ever applied, and it is now strict in
both directions: exact `Rule` and exact `FeatureTest` type identity (a subclass
whose `holds` returns `shell.family == "depth-outliers"` is refused, not run), a
refusal for an overridden `holds`, for a callable or non-finite threshold and
for a name outside the closed `FEATURES` table, and the full observable
consistency validator on every shell before a rule sees it. `declared()` now
compares the **complete structural rule body**, so a body edited after a result
cannot reuse an accepted `rule_id`.

## Identity: shells, records and realizations

* `ShellTruth` refuses a label whose boolean is not exactly
  `worst_analytic_distance > accuracy_tolerance`, and refuses a non-finite
  measurement.
* `ShellRecord` binds exact types, matching shell ids, matching face counts and
  a face set that is unique, non-negative, of the declared size and identical to
  the observable record's own.
* **Realization identity is a reconstruction-input digest**, not a frame digest.
  It hashes the ordered raw bytes, the fused poses actually used, the
  intrinsics, the window, the depth scale and bounds, the volume box, the cell
  parameters **and the backend the arm actually fused on**. A raw-only change, a
  pose-only change, a backend-only change and an exact clone are four separate
  controls; the run refuses two populations that share a digest, and refuses
  them loudly when they are on opposite sides of the split.
* The backend is in the identity because it can change a reconstruction: the
  shipped appliance cell is about thirty times `PYTHON_BACKEND_MAX_VOXELS` and
  runs on `numpy` while every declared grid cell runs on the pure-Python
  reference. `ObservableInputs.fusion_backend` is read off the volume the
  surface actually came out of rather than re-derived from the policy, and
  `check_inputs` refuses a backend this oracle does not run.
* The fusion memo in `tests.surface_oracle` was corrected to bind the same
  inputs. Its key used to name only the case, cell id, voxel size, backend and
  raw frame digests, so two fusions differing only in pose could collide.
  Cold-cache and warm-cache parity tests now show `fuse` and
  `fuse_keeping_volume` produce identical vertices, faces and `FieldRecord`, and
  that the weights under a shell come from the field that made its mesh.

## One bounded decision matrix

Round one applied all 195 rules to eleven overlapping confusion slices and then
again inside the selector: **89,505 invocations published as 16,380**, against a
ceiling of 40,000. That claim is withdrawn and is not repeated anywhere.

This round preflights `195 × (unsupported record count)` against the unchanged
40,000 ceiling, then evaluates every candidate on every declared unsupported
observable **exactly once** into an immutable cached `DecisionMatrix`. Every
confusion, every cross-tab cell, every aggregate, the eligibility test, the
ordering, the holdout gates, the recommendation and the renderer are lookups
into it. Every application of a rule anywhere is counted into a named phase, and
a test asserts the producer's count is exactly candidates × records and that
computing all 195 cross-tabs, all aggregates and the selection afterwards adds
**zero** further invocations.

## The population the sealed run produced

27 members at 2 resolutions = **54 populations, 54 fusions**.

| | value |
| --- | --- |
| closed shells derived (all) | 776 |
| unsupported closed shells (the labelled population) | 140 |
| supported closed shells (never labelled) | 636 |
| false shells / faces | 117 / 2,224 |
| correct shells / faces | 23 / 476 |
| training records (all resolutions) | 84 |
| sealed holdout records (all resolutions) | 56 |

At the **appliance** resolution, which is the only one selection and the gates
may read:

| split | records | false faces | correct faces |
| --- | --- | --- | --- |
| train (19 development members) | 32 | 712 | 316 |
| sealed holdout (8 members) | 41 | 624 | 152 |

The sealed appliance holdout carries both classes, which is the precondition for
it grading anything. Had it not, `run` refuses the population and returns it for
planner review rather than widening a magnitude until it passes. That refusal is
implemented and it does fire: the first execution of this round hit it, because
a defect in the new truth-free input guard had failed every population, and the
run stopped with "0 false and 0 correct" rather than reporting a holdout it did
not have. The guard was fixed; no member, magnitude, seed, candidate body, gate
or ceiling was touched.

## The result: no rule qualifies

195 predeclared candidates over 140 records = **27,300 producer rule
invocations**, one per cell, against the 40,000 ceiling.

* **Eligible rules: 0 of 195.** Reason recorded in the artifact: *"no declared
  rule removed at least 50% of the training split's false shell faces at
  appliance resolution while removing none of its correct ones."*
* `selected: null`, `holdout_confusion: null`, `gate_results: {}`,
  `counterfactuals: []`, `failures: []`, `decision: pass`,
  `recommendation: no-pruning-keep-dots-primary`.

Because no rule qualified, **no holdout gate was evaluated and no counterfactual
was run**. That is not a gap in the evidence; it is what the protocol does when
training selection produces nothing, and it is why the sealed holdout's labels
never touched the choice.

Selection reads the appliance cell alone. Round one aggregated the coarse
`v10-t4-w1-uniform` diagnostic cell with the appliance cell and so answered a
question the product does not ask; that cell is still fused, still published in
every cross-tab cell, and can no longer help a candidate qualify or pass.

Every candidate is published over the **exact split × perturbation-family ×
resolution cross-product**: 2 × 7 × 2 = **28 cells**. Only 8 of them hold a
shell — the `depth-outliers` and `combined` families, on both splits at both
resolutions — and the other **20 are published as zero cells** rather than
omitted, which is what stops a rule that only works on one family from looking
like a rule that works. Eleven readable aggregates are published beside them.
All 39 are recomputed by the strict reader from the records and its own audit
matrix.

## The truth-fed upper bound was predeclared, and could not be measured

The planner predeclared the upper bound as `sealed-outlier-185-p43` at appliance
resolution before that member had ever been built. When it was built, it
produced **no closed shell at all** at that resolution — so it has no false
unsupported shell, and there is nothing a perfect prune could have removed.

The artifact says so: `measured: false`, with the reason recorded. The member was
**not** swapped for one that would have produced a number. A predeclaration that
only binds when it flatters the result is not a predeclaration.

What a perfect prune is worth is therefore only available as **development/pilot
evidence**, measured on the training member `outliers-140-p17` at appliance
resolution with an explicitly truth-fed evaluator-only selector that is never a
candidate rule. This corrects round one's documentation, which reported the
correct unsupported faces as 0 before and after when the artifact said 92 and 92:

| measurement | before | after |
| --- | --- | --- |
| faces | 43,153 | 43,053 |
| unsupported false shells / faces | 3 / 100 | 0 / 0 |
| unsupported **correct** shells / faces | 2 / **92** | 2 / **92** |
| analytic vertex accuracy | 0.9686193317906873 | 0.9708263823900395 |
| triangle interior accuracy | 0.9784333263813254 | 0.9807059515790615 |
| reference coverage | 0.9999235512970797 | unchanged |
| protected area | 0.04217759009917748 | unchanged |
| bridging faces | 108 | unchanged |
| edge / thin-feature coverage | 1.0 / 1.0 | unchanged |
| worst dimension error | 0.0 | unchanged |
| non-manifold / boundary edges | 0 / 5,877 | unchanged |

That is **+0.2207 percentage points** of vertex accuracy across 100 of 43,153
triangles (0.2317 %), on one development member, with a selector that reads the
answer. It is the ceiling on the whole idea and no observable-only rule came
near it.

## Counterfactuals: no ceiling, and no prefilter

Round one capped counterfactuals at four (member, resolution) pairs and ran them
over the evaluator's *unsupported* subset. Both are gone.

* A rule that qualifies is applied to **all observable closed shells** — the 776
  of them, not the 140 — because an appliance cannot enumerate the unsupported
  ones. Anything analytically acceptable it removes, supported or unsupported,
  is **damage**: it is counted as `removed_supported_shells` /
  `removed_supported_shell_faces` and gated by
  `counterfactual_removes_no_supported_faces`.
* It is remeasured on **every** sealed holdout appliance population where it
  fires. There is no member ceiling; a partial sample can never support a
  recommendation.
* That list is an **exact ordered set**, and the strict reader derives it for
  itself rather than reading it: the sealed holdout populations at the shipped
  resolution on which applying the rule to every closed shell selects at least
  one. An omission, a duplicate, a training population, a diagnostic-resolution
  population and an invented population are each refused for their own reason.
  A population's split comes from the corpus, so a sealed holdout population
  whose closed shells the evaluator labelled **none** of still owes its
  counterfactual when the rule fires - and removing one of those shells is pure
  damage, which is exactly why it may not be skipped.
* `check_shell_identity` recomputes the mesh's connected components from
  `result.faces` and verifies each record's member, resolution, shell index and
  exact face set **before a triangle is removed**. A record pointing at another
  component is refused rather than measured — round one trusted the indices the
  record carried, so its advertised wrong-shell control happily measured the
  wrong triangles.
* Sixteen gates are named once in `GATES` and evaluated on the recomputed mesh:
  false faces fell, correct faces held, no supported faces removed, analytic and
  triangle accuracy held, reference coverage, protected volume, bridging, edge
  coverage, thin features, dimensions and topology.

## A reader that rebuilds the corpus, and binds the report to it

The previous pass's reader checked a report against *its own claims*. That is a
consistency check, not evidence, and the review proved it: fifteen mutations of
the published nine-megabyte artifact — another syntactically valid manifest
digest, another population digest, another base case, another member frame
digest, a coherently moved perturbation, another population `kind`, a
population's voxel work raised with the aggregate to match, the same for the
triangle work, counterfactual rule calls raised with the total to match, a
fabricated gain added to the unmeasured upper bound, every weight statistic and
the extraction threshold for one shell doubled so the published ratio survived,
one shell's vertex count and every ray count doubled so every published fraction
survived, a whole realization renamed to a fabricated reconstruction digest, an
invented extra realization, and a re-ordered member list — were all **accepted**.

`validate_report`, `loads_report` and `read_report` now establish the report
instead. They take **no evidence root**: a report is a claim about the corpus
this repository committed, so the reader loads that corpus itself with
`load_corpus()`, rebuilds every declared member at every declared resolution
from the committed bytes, derives every closed shell and its label, and binds
the document to that rebuild. (`load_corpus(root)` remains the named API for
checking a copied or corrupted fixture root; it is not the strict reader.) The
rebuild is memoised once per process under a key made only of committed digests
and code-level protocol identity — including the fusion policy the populations
are actually produced on, evaluated over every declared resolution rather than
named beside it, so moving the diagnostic arm off the reference backend and
onto the appliance one is a different entry although no committed byte moved.
A producer run and a strict read in the same process pay for the rebuild once,
and a second corpus can never reach the first one's entry.

The reader then does two different things, and they close different attacks.

**It binds.** The corpus block field by field — the manifest, population and
base-case digests, the candidate-only and protocol digests, the resolutions and
their order, the selection resolution, the material fraction, the work ceiling,
the sealed list and its order, the predeclared upper bound, and every member's
exact ordered id, family, split, seed, control case, frame digest and
perturbation. Then every realization as an exact ordered set; then every
population — schema, kind, member metadata, voxel size, reconstruction digest,
counts, unavailable items, voxel work, triangle samples, every closed shell and
every labelled record — against the population the corpus rebuilds. A coherent
forgery is refused here because *the evidence does not contain it*.

**It recomputes.** Every typed `ObservableShell`, `ShellTruth` and `ShellRecord`
is rebuilt and run through the module's own invariants; every closed shell
identity must be globally unique, labelled or not; one **independent audit
decision matrix** is built — accounted separately as `reader_audit_rule_calls`,
never hidden as producer search — and the 28-cell cross-tab, all eleven
aggregates, the eligible ordering, the selected rule and its stated reason, the
holdout confusion, every gate, the recommendation, every total and the decision
are recomputed from it. The counterfactual list is derived rather than read (see
above) and each entry's identity is bound to its canonical population before its
deltas, damage breakdown, twelve gate booleans and `passed` status are
recomputed - and then the whole entry is bound to a **re-measurement**: the
reader re-fuses that member, prunes exactly those shells and re-measures the
mesh with the accepted oracle's own code, so a `before` / `after` pair that
agrees with itself perfectly but with no mesh is refused. The upper bound's
**form** follows the predeclared population rather
than a boolean in the report: measured and every number recomputed if that
population carries false unsupported shells, and otherwise exactly six keys, an
empty removal list and the canonical reason — which is why a fabricated
`false_faces_removed` on an unmeasured ceiling is now refused as an unexpected
key. Every line of the work account — fusions, voxel visits, triangle samples,
producer calls, counterfactual calls, reader calls, unaccounted calls, the
total, the matrix size, the ceiling, the remeasurement count and the exact
explanatory note — is recomputed from the rebuild and from what the protocol
actually spends.

Every scalar goes through a typed accessor, because `True == 1` in Python and
`str(item)` turns anything at all into a string: a boolean cannot satisfy an
integer schema field and a non-string cannot stand in for a list item.

Binding a value to the rebuilt one does not remove that need, it moves it:
`==` does not distinguish `False` from `0`, or `True` from `1` and `1.0`. So
the two places where a whole mapping was compared rather than read field by
field are typed first. Every field of a member's **perturbation** — the family
label as text, the depth, outlier and band settings as strict integers, the
pose magnitudes as finite numbers, the translation bias and rotation axis as
finite three-vectors, and the key set exactly — is established before the
comparison, and both **surface accounts** of every counterfactual and of a
measured upper bound have every count typed as an integer and every other
measurement as a finite number before a delta, a gate, a `passed` status or the
re-measurement binding reads any of them.

The controls: **16** over the committed corpus, one over the trusted rebuild's
cache key alone — a backend-only fusion-policy change has to change it, proven
without fusing anything — **67** adverse controls over a small built report
that exercise the derived logic a negative sealed run never reaches (a
qualifying rule, its counterfactuals and a measured ceiling), and **23** adverse
controls that mutate the real committed artifact and hand it to the public
boundary: one for each of the fifteen forgeries above, four type confusions an
exact comparison inherits (a boolean where a sealed integer, a pose magnitude
or an axis component belongs, and a perturbation field simply removed), a
swapped predeclared bound, an unmeasured bound claiming to be measured, a
rewritten unmeasured reason and a forged failure list. Each refuses for its own
reason.

## Computational cost

Deterministic work, published in the artifact; wall clock lives here because an
artifact that must be byte-identical between two runs cannot carry a number that
differs between them.

| | |
| --- | --- |
| fusions | 54 |
| voxel visits | 263,867,275 |
| triangle samples | 14,693,751 |
| producer rule invocations | **27,300** (195 × 140, one per cell) |
| counterfactual rule invocations | 0 (no rule qualified) |
| reader-audit rule invocations | 0 in the producer; 27,300 when a standalone reader audits the artifact |
| unaccounted rule invocations | 0 |
| declared ceiling | 40,000 |
| counterfactual remeasurements | 0 |

The command took **6 m 02.0 s** and **6 m 05.7 s** on the recorded host, over two
consecutive runs. The artifact is 9,213,803 bytes, SHA-256
`f40f2c5d312cf722217d21447490c05bcf50899161f5ddd2bacd5b8a19ff3767`, and the two
runs produced byte-identical files.

A **cold standalone strict read** of that artifact — a fresh process that has
the report and the corpus and nothing else — took **367.7 s**, almost all of it
the trusted rebuild, and its rule-call ledger came out as
`{'reader_audit': 27300}`: one independent audit decision matrix, no producer
search and nothing unaccounted. A second read in the same process took **0.3 s**
against the memoised rebuild and took the ledger to 54,600. That is the price
of a reader that establishes its evidence, and it is paid once per process
rather than once per report.

The digest moved from round two's `2eeac220…` for one reason: the fusion
backend is now inside the reconstruction-input identity, so every realization
digest in the artifact changed. Nothing else did — the same 27 members, 54
populations, 776 closed shells, 140 records, 27,300 decisions and the same
negative result.

## Physical evidence: there is none, and that is the report

Both named read-only replays stop before fusion and say so.

| capture | files | frames | stopped at | read-only |
| --- | --- | --- | --- | --- |
| `tests/fixtures/turntable` | 25 | 24 named, 24 used | registration: frame 3 cannot be placed (adjacent 21.994, keyframe 24.158, both over the 15.600 gate) | unchanged, confirmed |
| `/home/ben/scans/scan/20260828-110641` (limit 12) | 166 | 166 named, 12 used | registration: frame 1 cannot be placed; bounded coarse recovery refused too | unchanged, confirmed |

`shell_population_available` is `false` on both, with the reason recorded. No
physical shell, discrimination or accuracy claim exists, and none can while
registration and readiness stop before fusion.

The replay module no longer carries a fusion arm at all. Round one kept an
untested branch past the readiness gate that would have used the Python backend
while the experiment's appliance arm uses NumPy; it has been removed rather than
left to imply a comparison it could not make. Every exit path — normal return,
a refusal, any other `ScannerError`, any other exception — now takes the
after-fingerprint, compares it and lets a changed capture dominate the exit code.

## Limitations

1. **Synthetic.** `scanner3d.simulate` is not a sensor model. A separation
   demonstrated here would be a separation of the arithmetic on exact truth:
   necessary evidence for a product change, never sufficient.
2. **The labelled population is not one an appliance can enumerate.**
   "Unsupported" is defined against the scene's ideal returns, which is
   evaluator knowledge. That is why a qualifying rule is applied to all 776
   closed shells rather than to the 140, and why any acceptable surface it
   removes is published as damage.
3. **The predeclared upper bound is unmeasured.** The sealed member the planner
   named produced no closed shell at appliance resolution. The ceiling quoted
   above is development/pilot evidence from a training member.
4. **No physical shell population exists.** See above.
5. **Nothing here is wired into `scanner3d`.** There is no pruning code path in
   the product, this experiment does not add one, and a recommendation from it
   would still be an owner decision with its own task.

## The next action

Keep dots primary and add no pruning.

The honest follow-up is **not** a wider threshold search on this synthetic
corpus: the sealed run says the observable vocabulary as declared does not
separate the two classes at the resolution an owner actually gets, and widening
the grid after seeing that is the exact mistake round one made. The follow-up
that would change the answer is a **physical** shell population, and that needs
the registration/readiness wall to move first — both recorded captures refuse
before fusion.
