# The reconstruction quality harness

Status: added 2026-08-28; the shipped fixture corpus and its CI job added
2026-08-29. The fast CPU correctness tier is in place, is load-bearing, and now
runs in public CI as its own named job with a preserved report; the opt-in
physical replay tier reports but does not yet gate, because the physical truth
it would gate against has not been measured. Both run offline and need no
scanner, no display server, no GPU and no network, and the default compute
backend is the pure-Python reference.

The measurements live in [`src/scanner3d/quality.py`](../src/scanner3d/quality.py)
with their regression cases in [`tests/test_quality.py`](../tests/test_quality.py).
The known geometry also ships as data, under
[`tests/fixtures/quality/cases/`](../tests/fixtures/quality/cases), run by one
command - see [The fixture corpus](#the-fixture-corpus-and-the-one-command-that-runs-it).

## Why this exists

Until now a scanner change was judged two ways: the pipeline's own numbers, and
whether the dots looked right to the owner. Both are real, and neither is
enough.

`scanner3d.pipeline.reconstruct_sweep` reports **held-out closure error**, which
is an honest check and a hard-won one — it is scored against revisit
constraints deliberately withheld from the fit, so it is not training error.
But it is a statement about *self-consistency*: the poses agree with each
other. A reconstruction can be perfectly self-consistent and be the wrong
shape, at the wrong scale, with a third of the object missing.

On **2026-08-28** that stopped being hypothetical. A relocalization bridge was
accepted at residual 0.515 with a 1.50x refined runner-up margin and a matched
fraction of 1.00 — every internal number healthy — and the placement it chose
was wrong. Its own record said `coarse_unambiguous: false`, and a genuinely
different refined candidate covered *more* of the model (0.598) than the winner
did (0.483). `scanner3d.relocalize` now refuses an ambiguous coarse search
outright. This harness is the other half of that fix: the machinery that
notices when a change makes the reconstruction worse, before a scan does.

## Separate failures, separate instruments

There is deliberately **no single quality score**. Different failures need
different instruments, and one number cannot say which one happened.

| Measurement | Catches | Blind to |
| --- | --- | --- |
| Held-out closure (`scanner3d.pipeline`) | poses that disagree with withheld revisits | any error the whole reconstruction shares — scale, a global misplacement, a missing side |
| `pose_sequence_error` | the camera being in the wrong place, per frame, against exact truth | anything that is not a pose |
| `cloud_agreement` (both directions) | displaced, lumpy, spurious or **missing** surface; a uniform scale, when the reference is dense and in the same frame | a wrong placement of a symmetric object; anything the supplied reference does not cover |
| `plane_roughness` | a surface known to be flat coming out bumpy | scale and shear — a shear maps planes to planes |
| `dimension_errors` / `landmark_offsets` | the right shape at the wrong size; a feature not where it belongs | anything nobody supplied truth for |
| `audit_bridge_record` | a historical relocalization decision current policy would refuse | geometry — it re-judges recorded numbers, it does not re-run the search |

The important pairings:

- **Closure and cloud agreement answer different questions.** Closure is
  computed inside the reconstruction's own coordinate system. A world scaled by
  2% has identical pose rotations, so the closure score — which is the rotation
  residual of the withheld edges — is *literally unchanged*. Cloud agreement
  against real truth is not.
- **Cloud agreement must be symmetric.** Observed-to-reference is blind to a
  hole: every point that exists can sit exactly on the truth while half the
  object is absent. Reference-to-observed is blind to geometry that should not
  be there at all. Reporting either one alone passes a reconstruction the other
  fails, so both are always computed and both are always reported.
- **Roughness sees what a whole-cloud average does not.** A distance summary
  over a whole model dilutes a locally bumpy panel into nothing. A plane fit on
  a region the caller says is flat does not.
- **A physical unit comes from the truth, not from the instrument.** Every
  length here inherits the unit of the truth it is given: a dense reference
  surface expressed in millimetres makes `cloud_agreement` a millimetre
  distance, and exact poses with a named translation unit do the same for
  `pose_sequence_error`. What is missing here is that truth, not the
  capability. This repository's physical fixture has no dense truth surface and
  no exact poses, and the physical meaning of an arbitrary reconstruction's own
  coordinates is still not established, so those measurements are reported in
  `COORDINATE_UNIT` today. Independently measured landmarks and dimensions are
  the physical truth that is presently practical to record here, which is why
  they are what the calibration step below asks for.
- **Scale needs truth from outside the reconstruction.** A uniform scale error
  is invisible to held-out closure, to plane roughness and to every other
  self-consistency check, because all of them are computed inside the
  reconstruction's own coordinate system and a 2% larger world is perfectly
  consistent in it. `cloud_agreement` is not one of those: it measures against
  a reference the caller supplies, so a *dense* reference surface in the same
  frame does see a 2% scale — it is 2% away everywhere. Such a surface rarely
  exists for a physical object, and where it does not, named dimensions and
  landmarks are the sparse physical check that remains. That is a statement
  about which truth is available, not about which instrument is capable.

## Gates belong to the caller

The module owns no notion of "good". Every threshold is supplied at the call
site through `at_most`, `at_least` or `must_be`, in the same unit as the
measurement it judges, with a sentence saying what to *do* when it fails:

```python
quality.at_most(
    "cloud.reference_to_observed.p95",
    agreement.reference_to_observed.distances.p95,
    0.10,
    unit=quality.COORDINATE_UNIT,
    context="truth surface has no reconstruction near it; capture the missing side",
)
```

A named limit next to the case it governs can be argued about by someone who
can see the argument. A constant inside the module would be a hidden standard
nobody voted on.

`Gate` is a public frozen dataclass, so its invariants live on the class rather
than in those three factories: a gate built by calling `Gate(...)` directly is
validated identically — a non-empty name, unit and action; a comparison the
module actually defines; finite, non-boolean numbers under an ordering; real
booleans under `is`. A comparison it does not recognise is refused at
construction instead of falling through to an ordering and reporting a PASS for
a question nobody asked.

`CaseReport` and `SuiteReport` evaluate **every** gate and list **every**
failure — never stopping at the first — because the second failure is usually
what says which failure mode actually happened. Both render deterministically
for a CI log and round-trip through strict JSON (`allow_nan=False`) on the
failing path as well as the passing one.

## What the fast tier proves today

`tests/test_quality.py` runs in 0.84 s on one core of the machine this was written on,
interpreter startup included. It proves:

1. **A clean case passes.** A known moving camera over a known surface with a
   real but small pose error (0.01 degrees and 0.01 units per frame,
   accumulating) clears every gate. The harness is not a reject-everything
   stub.
2. **The gauge is removed without removing the error.** Two pose sequences
   differing by one global transform report zero; the same sequences with a
   per-frame divergence report all of it. Nothing is fitted to the
   trajectories — a least-squares alignment over all frames would absorb part
   of the very error being measured. Frame 0's error is zero by construction
   and is excluded from the summaries it would otherwise flatter.
3. **A subtle lump trips its gate.** A ripple 0.30 high on a panel 60 across —
   half a percent, invisible on a rendered model — fails a roughness gate the
   same panel without it clears by twelve orders of magnitude.
4. **An incomplete cloud cannot pass on one direction.** Every surviving point
   is byte-for-byte a point of the truth, so observed-to-reference is exactly
   zero with coverage exactly 1.0. The missing cap fails the reverse direction
   and the reverse coverage gate.
5. **A wrong placement of a symmetric object is only visible in pose truth.**
   A square prism that maps onto itself under an exact quarter turn is
   reconstructed with half its frames a quarter turn out. Both cloud
   directions are exactly zero — a flattering surface residual over a
   placement that is simply wrong — and only the comparison against exact
   poses catches it. This is the 2026-08-28 failure in miniature.
6. **A uniform scale trips the dimension and landmark gates.** 2% larger:
   rotation error exactly zero, plane residual below 1e-12, and the dimension
   and landmark gates the only two of the four this case runs that fail. The
   case deliberately runs no `cloud_agreement`: a dense reference surface in
   the same frame would see the scale too (see above), and the point being made
   is that the sparse named-length check catches it when no such surface is on
   hand — which is the situation this repository is actually in.
7. **The 2026-08-28 bridge stays refused.** The record is preserved byte-for-byte
   under `tests/fixtures/quality/` with its source path and SHA-256, and the
   test binds that hash. It is refused *specifically* because current policy
   forbids an ambiguous coarse search — asserting only "refused" would keep
   passing if the refusal ever came from somewhere else.
8. **A healthy bridge is still accepted.** Built through the real
   `scanner3d.relocalize` types and `as_record`, so the audit is proved to read
   the schema the product actually emits rather than a hand-written imitation.
9. **Every re-applied policy gate fires on its own.** Each of the eight gates
   is tripped individually and asserted to be the only refusal.
10. **The accelerated backend agrees with the reference.** `scipy` agrees
    exactly; `numpy` agrees to 7.5e-9 of the scene's own extent, which is where
    `BACKEND_AGREEMENT` came from — see below.

Unusable input is refused before anything is allocated: empty and non-finite
clouds, mismatched pose sequences, mismatched landmark names, a non-positive
coverage tolerance, more than `MAX_POINTS` on either side, and a comparison
whose cost exceeds `MAX_NEAREST_WORK` even when each side is individually
within bounds. Gates are refused the same way whichever route built them: every
invalid direct `Gate(...)` form has its own regression, alongside a valid one
that is asserted equal to its factory-built twin.

## The fixture corpus, and the one command that runs it

Everything above lives in Python. That is enough to prove the instruments
work, and it is not enough to be a regression harness: a case that exists only
as a helper function is a case whose truth nobody can read, diff or review
without reading the code that generates it.

So the known geometry is also **data**, under
[`tests/fixtures/quality/cases/`](../tests/fixtures/quality/cases), and one
command runs it:

```bash
.venv/bin/python -m tests.quality_harness --out quality-harness-report.json
```

- **`0`** - every case produced exactly the outcome its fixture declares.
- **`1`** - the harness ran and at least one case did not. An unexpected pass,
  a missing expected failure, an extra failure, a malformed fixture, an
  unusable measurement and an exception while measuring are all this.
- **`2`** - the harness could not run at all: no corpus, a manifest that is not
  a manifest, a case file that is not where the manifest says, or a report that
  could not be published.

Independent cases continue after a failure, so the artifact preserves *all* of
them. The first broken case is rarely the informative one.

### What is in the corpus

Ten cases, each a single JSON file stating its own truth, its own gates with
their units and limits, and the exact outcome it must produce:

| Case | Declares | And proves |
| --- | --- | --- |
| `clean-surface` | passes | the harness is not a reject-everything stub |
| `clean-panel` | passes | the roughness instrument's own control |
| `missing-surface` | fails **only** the two reference-to-observed gates | every surviving point is byte-for-byte a point of the truth, so the forward direction is exactly zero at coverage 1.0 |
| `spurious-geometry` | fails **only** the two observed-to-reference gates | the mirror image: nothing of the truth is missing, and the invented lobe is visible one way round only |
| `lumpy-panel` | fails both roughness gates | a ripple half a percent of the panel, invisible on a rendered model and diluted to nothing by a whole-cloud average |
| `uniform-scale` | fails the dimension and landmark gates, and **passes** the relative-error gate beside them | a 2% world is perfectly self-consistent; only truth from outside the reconstruction sees it |
| `readiness-valid-orbit` | the real reader accepts it | the readiness control |
| `readiness-poses-stationary` | `pose_viewpoints_insufficient`, `pose_excursion_insufficient`, `pose_motion_below_noise`, `coverage_unsupported` | known-invalid pose evidence, refused by its own reason codes |
| `readiness-coverage-one-sided` | `coverage_unsupported`, **and nothing else** | known-invalid occupied coverage, firing alone |
| `readiness-surface-displaced` | `surface_unsupported_by_evidence` | known-invalid surface evidence, refused after fusion |

**A bad case is a harness PASS only when the exact expected set of gates
fails.** Every cloud case states all four cloud gates and every readiness case
names every reason code the authority defines — present *or* absent — so a
case cannot be satisfied by failing everything, and "refused for some reason"
is never the refusal a fixture was written for. That is what
`tests/test_quality_harness.py` corrupts in eleven different ways to prove.

The readiness records are not hand-written. They are the output of the real
`evaluate_readiness` and `surface_evidence`, frozen, and read back by the real
`ReconstructionReadiness.from_dict` — no rule of that module is restated here.
`test_the_frozen_records_are_what_todays_evaluator_still_produces` re-runs the
evaluator over the same synthetic sweep, so a record the product would no
longer produce cannot sit in the corpus pretending to test something.

**Limitation, stated rather than hidden:** the readiness tier exercises the
strict record boundary, not a full export. A fixture carrying real raw frames
and a real fusion would add megabytes and seconds to every CI run to re-prove
what `tests/test_reconstruction_export_readiness.py` already proves about the
export path; what the corpus measures is that the gates and their reason codes
still fire on the evidence they were written for.

`tests/build_quality_fixtures.py` regenerates the corpus deterministically and
is the reviewable recipe behind those coordinates. It is a command, not a test:
pytest never runs it, and
`test_the_generator_reproduces_the_shipped_corpus_exactly` checks the shipped
bytes against it in a temporary directory rather than rewriting the evidence to
make itself pass.

The byte-bound 2026-08-28 bridge record is untouched beside the corpus, in
`tests/fixtures/quality/` with its own schema-1 manifest. It is source
evidence and was not migrated into the new schema.

### The artifact

One strict JSON document at **report schema version 2**, published atomically
— serialised in full, written to a temporary file in the same directory,
fsynced, then renamed — so a reader never sees half a report and never sees a
stale one presented as current. It carries its schema, kind and version, the
corpus manifest digest, each fixture's name, size and SHA-256, the expected and
observed outcome, every measurement and gate, each gate's **binding** to the
measurement and path its value came from, the per-case status, aggregate
counts, the overall decision and the limitations above.

Version 2 is what added those bindings and the exact per-kind measurement
schemas below. A version-1 reader accepts a measurement this one refuses, so
the number moved rather than the meaning of the old one; a consumer pinned to
version 1 is reading a weaker promise than the same field name suggests.

It is written through the same reader that reads it back, so an artifact this
harness could not itself accept is never published.

That reader believes nothing the report asserts about itself where the report
also carries what the assertion is derived from. A gate's verdict is
recomputed from its own measured value, comparison and limit; a case's
`passed`, `gate_count` and `failures` from its gates; its observed outcome and
failing set from those same gates; its mismatch prose and its status from
expected versus observed; and the totals and the decision from the cases. On
top of that it refuses an unknown or missing field at any depth, an unknown
schema or kind — of the report, of the corpus it names or of the embedded
quality report — a fixture name that is not a plain corpus file name, a digest
that is not a SHA-256, a boolean standing in for a number, a non-finite value,
a duplicate case id, gate name or failure, a non-canonical expected/observed
block, an entry that says it was malformed while carrying measurements, totals
that do not add up and a decision its own totals contradict.

**Every embedded measurement is held to the exact schema of its own kind** —
`cloud_agreement`, `plane_roughness`, `dimension_errors`, `landmark_offsets`
and `readiness_record`, which are all five of the kinds this harness emits, and
a kind it has no schema for is refused rather than skipped. Exact nested keys
throughout, so an unknown field added to a measurement and a required one
removed are both refusals; natural types, so a boolean cannot stand in for a
number; and the invariants each kind's own construction guarantees. Those
invariants are derived from what the artifact already carries and nothing else:
a distribution summarises exactly the population its measurement reports, with
median ≤ p95 ≤ worst and an rms no larger than the worst magnitude; a coverage
fraction is a whole number of the query points it counts; the two directions of
a cloud agreement agree on how big each cloud is, share one tolerance and stay
inside the search gate their neighbours were found in; a plane normal is a unit
vector and its worst residual is the larger of its signed extremes; a dimension
entry's error, absolute error and relative error follow from its own measured
and truth values, with the relative summary present exactly when every truth is
non-zero; a landmark's offset is the distance between the two points beside it;
and a readiness record's `ready`, `reason_count`, `reason_present` flags,
canonical reason order and one-line summary all follow from its reason codes.
No geometry is recomputed that the artifact does not contain, and no
independent truth is invented.

**Each gate is bound to where its value came from.** `scanner3d.quality.Gate`
records a measured value and nothing about its provenance, and that schema
belongs to the authority — so the binding lives beside the gate in the
artifact instead: `gate_bindings` gives one `{gate, measurement, path}` entry
per gate, in the gates' own order, naming the fixture-declared measurement and
the path into that measurement's serialised form. The reader resolves the path
itself and requires the gate's `measured` value to be the value found there, as
the same kind of thing, so a boolean cannot stand in for a number. It refuses
an unknown measurement name, an empty path, a path step that is not a plain
field name, a path into a field the measurement does not have, a duplicate
binding, an unbound gate, a binding for a gate the case does not report,
bindings out of the gates' order and a malformed case carrying any binding at
all. This is what catches a measurement changed while the gate that read it was
left alone — the one forgery a report carrying only gates has no way to see.

Read a published artifact with `read_report(path)` or `loads_report(text)`,
not by handing `json.loads` output to `validate_report`. Those two parse the
text themselves and refuse a duplicate JSON key and the non-standard `NaN` and
`Infinity` constants, neither of which any validator over an
already-decoded mapping can see: Python's decoder keeps the last of two
identical keys and silently discards the first. Serialisation is
`allow_nan=False`.

**A requested output path never keeps an earlier answer.** The destination and
its staging file are cleared, by exact name, *before* the corpus is opened — so
a run that refuses at exit 2, a document that will not serialise and a write or
rename that fails all leave no file rather than the previous run's plausible
report, which carries a real digest, real cases and a real decision and says
nothing about not being this run's. A destination that is a directory or a
symbolic link is refused rather than removed; neither is this harness's
artifact.

**Two runs over an unchanged corpus are byte-identical.** No duration, host,
path or environment appears anywhere in it — a number that changed with the
machine would make "the report changed" useless as a signal.

### In CI

`.github/workflows/ci.yml` runs it as its own job, `Reconstruction quality
harness`, on public GitHub-hosted `ubuntu-latest` with `contents: read`, a
bounded timeout and every action pinned to a reviewed full commit SHA. The step
prints the whole report to the log — every gate, passing ones included — and
the upload step is guarded by `if: always()` with `if-no-files-found: error`,
because the failing run is the one whose report is worth keeping and a missing
report must be loud rather than quiet. `tests/test_ci_workflow.py` parses the
workflow and asserts all of that, including that no job reaches for hardware or
for the opt-in physical replay tools.

Ordinary pytest still runs, unchanged, in its own job. The harness is
**additional** named evidence, not a replacement for it.

It is one of **three** named CPU correctness-oracle jobs, and they are not
interchangeable. This one, `Reconstruction quality harness`, runs
`tests.quality_harness` over `tests/fixtures/quality/cases/` and keeps
`quality-harness-report`. The second, `Depth-completeness benchmark`, runs
`tests.depth_completeness_harness` over
`tests/fixtures/depth_completeness/cases/` and keeps
`depth-completeness-report` — see
[docs/depth-completeness-benchmark.md](depth-completeness-benchmark.md). The
third, `Surface regression oracle`, runs `tests.surface_oracle_harness` over
`tests/fixtures/surface_oracle/cases/` and keeps `surface-oracle-report` — see
[docs/surface-oracle.md](surface-oracle.md). They share this module's
path-safety, strict reading and atomic publishing, but they measure different
evidence against different schemas, so folding them into one job would hide
which of the three went red and let one report overwrite another. All three are
`ubuntu-latest`, CPU-only, and identical in runner, permissions, pinned actions
and frozen sync. All three corpora are also run for real inside ordinary pytest,
by the `test_*_harness.py` module beside each one; the named jobs exist for a
visible per-oracle status and a report that outlives the runner, not because the
suite omits them.

## Backends, and why "python" is the oracle

`cloud_agreement` searches through `scanner3d.compute`'s backend seam and
records which backend resolved. The default is `"python"`, so the correctness
tier does not depend on numpy or scipy being installed.

`BACKEND_AGREEMENT` is stated as a **fraction of the compared scene's own
size**, not as an absolute length, because that is the unit the disagreement
lives in. The numpy backend computes `|q-t|^2 = |q|^2 - 2 q.t + |t|^2` for
speed, and that identity loses about half the mantissa to cancellation exactly
where the true distance is near zero — which is where a *good* reconstruction
sits. Measured on the parity scene: scipy 0.0, numpy at most 7.5e-9 of the
scene extent (4.8e-07 absolute on a 64-unit scene, and only on the median).

**Timing is never a gate.** No report carries a duration: a report that did
would not be reproducible, and a duration is a property of the machine, not of
the reconstruction. Measure it beside the harness.

**A GPU runner is not the correctness oracle**, which is why the CI job above
is `ubuntu-latest` and nothing else. The corpus runs in well under a second on
one CPU core through the pure-Python backend, so a GPU runner would change how
fast the answer arrives and not what the answer is - and it would make the
oracle depend on a driver stack whose numerical behaviour is the thing under
test elsewhere. GPU belongs to two separate future lanes — OpenGL/viewer parity (`scanner3d.gl_cloud_view`,
`scanner3d.cloud_viewer`) and performance measurement — and neither decides
whether a reconstruction is right. `scanner3d.compute` already reserves a
`"gpu"` backend name that always reports unavailable; when it exists, its job
is to agree with `"python"`, not to replace it.

## The opt-in physical tier

Everything above is synthetic, and that is a limit worth stating plainly: it
proves the *harness* is load-bearing, and it proves a change did not break the
geometry the harness measures. It does not prove anything about the scanner.

The physical tier is opt-in, read-only, and deliberately not collected by
pytest, for the same reason `tests/tsdf_benchmark.py` is not: what it measures
depends on the capture and the machine, so asserting bounds on it in the suite
would produce a test that fails elsewhere and tells nobody anything.

`tests/quality_replay.py` is that tier:

```bash
.venv/bin/python -m tests.quality_replay                       # the shipped turntable fixture
.venv/bin/python -m tests.quality_replay --capture /path/to/capture
.venv/bin/python -m tests.quality_replay --register --frames 3 --max-icp-rmse 2.0
.venv/bin/python -m tests.quality_replay --plane-roi 0,50,0,50,150,220 --plane-name "bed floor"
.venv/bin/python -m tests.quality_replay --truth truth.json --max-dimension-error 2.0
.venv/bin/python -m tests.quality_replay --capture /path/to/capture --require-bridge-pass
```

The capture directory is opened read-only — every file is read with
`read_bytes`/`read_text` and nothing in the tool ever opens a path there for
writing. It touches no device node, no USB, and starts no capture. It prints
one JSON document to stdout — serialised whole, and with `allow_nan=False`, so
a non-finite value anywhere in the envelope refuses the document instead of
writing `NaN` into it — and exits **1** if any gate the caller asked for
failed, **2** if it could not run at all (a `--capture` that is not a
directory, a `manifest.json`, `--truth` file or `bridge.json` that is not the
shape it must be). Those two are kept apart so a caller can tell "this capture
failed a gate" from "this harness never measured anything".

A frame whose byte count is not the size this device records is set aside
before deprojection, named under `frames_set_aside` with the reason, and left
to the `frame_integrity` case to report and — under `--require-integrity` —
gate. One bad file costs that frame, not the whole report.

Measured on the shipped fixture with no flags, in about half a second:

- **frame integrity** — byte count and SHA-256 of all 24 frames, checked
  against the manifest's recorded hashes (24 of 24 match). A frame the
  manifest does not name reports `matches_manifest: null`, which is a
  different fact from a mismatch and is never conflated with one;
- **cloud statistics** — per-frame deprojected point counts and extents, with
  the depth clip and the downsample stride recorded in the report rather than
  assumed;
- **`not measured`** for everything else, each with the reason.

What it will not do:

- **No unaligned frame-to-frame "accuracy".** The 24 frames are 24 *different
  views* of a turning object. Their raw disagreement is motion, not error.
  Cloud agreement is offered only behind `--register`, which first aligns the
  consecutive pair with `scanner3d.registration.icp` and labels the result
  "agreement after alignment, not accuracy against truth" wherever it appears.
  Without the flag the case carries a `not measured` note instead.
- **No region chosen on the caller's behalf.** Plane roughness needs an
  explicit `--plane-roi` and `--plane-name`.
- **No invented "measured" side.** This repository has no automated
  named-feature extractor, so a `--truth` file states both the value the
  caller trusts as measured and the truth it is judged against.
- **No thresholds from manifest prose.** The refusal is a named constant,
  `MANIFEST_PROSE_IS_NOT_TRUTH`, quoted into the report whenever a
  truth-dependent case has nothing else to say.

Gates are opt-in and named after the flag that supplies the limit
(`--max-plane-p95`, `--max-dimension-error`, `--max-landmark-offset`,
`--max-agreement-p95`, `--max-icp-rmse`, `--require-integrity`,
`--require-bridge-pass`). A flag that is not passed adds no gate, and the
report carries `gate_count` beside `passed` on every case and on the suite —
so "nothing failed" and "nothing was asked" cannot be read as the same thing.

Wall-clock timing appears only under the tool's own top-level `"timing"` key,
never inside anything `scanner3d.quality` serialises.

## What is NOT measured yet, and what would fix it

`tests/fixtures/turntable/` is 24 immutable physical frames of a model truck
bed, and its manifest describes the object as "roughly 110 x 95 x 24 mm". That
is **prose, not dimensional truth**, and the harness must never turn it into a
threshold. Nobody recorded which faces are planar, where the fixture's
landmarks are, or what instrument produced those numbers, and the depth unit of
an arbitrary reconstruction's coordinates is still not established (see
[metric-validation.md](metric-validation.md) for what *has* been established,
and on which single target at which single standoff).

So the physical tier reports what the evidence supports and labels the rest
`not measured` — never `pass`. A landmark gate that silently does not exist
reads, in a report, exactly like a landmark gate that passed.

**A purpose-built physical target now supplies the missing nominal geometry.**
[`reference/targets/reconstruction_v1/`](../reference/targets/reconstruction_v1/)
contains a printable 92 x 82 x 56 mm asymmetric solid, a byte-bound dense CAD
surface, exact planar regions, landmarks, dimensions and a print/scan protocol.
Its planes, ramp, deep pocket, smooth bump, four steps and three resolution-scale
rails were chosen specifically to separate coverage, unsupported fill, edge
loss, dimensional error, lumpiness and camera-motion tracking.

That closes the *design* gap, not the physical-evidence gap. The committed STL
is nominal CAD truth until the individual print exists and is measured. Before
the physical tier can gate a scan of it, the owner still needs to provide:

- at least three caliper readings of the named overall dimensions, plateau
  height, pocket opening and pocket depth, with the instrument resolution and
  print settings;
- the two photographs named by the target protocol, so damage, brim cleanup and
  orientation are not silently forgotten;
- one raw controlled relative-motion scan at its recorded standoff — preferably
  a fixed camera and continuously rotating turntable for the first run —
  without moving both sides, flipping or repositioning the target.

The repository owns the coordinate frame, dense reference, ROI bounds,
alignment and gate construction; the owner does not need to prepare JSON or
choose thresholds. Until the actual print measurements and raw capture exist,
the physical tier remains a report rather than a gate. Exact per-frame pose
truth would additionally require an external tracker or calibrated motion rig;
this target makes a bad pose visible through CAD disagreement and closure, but
does not pretend to measure camera motion independently.
