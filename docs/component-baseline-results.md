# Measured-target baseline — 2026-09-09

## Outcome

The saved fixed-axis reconstruction reproduces the historical **approximately
0.50 mm CAD-surface result**, but explicit base dimensions are **0.8-0.9 mm
undersized** relative to the owner's calipers. This is a coherent, recognizable
shape, not a completed dimensional-reliability qualification. No physical repeat
has run, no scale correction has been applied, and general handheld tracking
remains unqualified.

Evidence directory:
`/home/ben/scans/diagnostics/20260909-component-baseline-WzP3OE/`.
The native FreeCAD document is `measured-target-overlay.FCStd`; reference-grey
and scan-cyan screenshots are `overlay-axonometric.png`,
`scan-only-axonometric.png` and `scan-only-top.png`. The CAD is the existing
reference STL with the predeclared print correction; the scan is points only.
No experimental scan mesh was generated.

## Execution and an evaluator bottleneck

- Input: all 190 original 512,000-byte frames in
  `/home/ben/scans/sweep/20260904-020845`.
- Settings: explicit 60-300 mm band, placement seed 5, fixed-axis path enabled.
  The registration selected SciPy, found 126.70 frames/turn and completed without
  falling back to the general frame chain. A new cache preserved its 389,785
  prepared/registered points and all poses; no old cache was used for registration.
- The unchanged command started at 17:25:08 UTC and reached the declared
  900-second limit at 17:40:09, after saving the cloud/poses and passing the revisit
  check but before emitting its final CAD report. Exit -15 is the timeout
  supervisor's termination, not a geometric refusal. Do not describe this as an
  unchanged end-to-end command successfully finishing.
- Inspection found full SVD calls whose callers use only the right singular
  vectors for plane normals. On the actual largest 38,039-by-3 face input,
  requesting the unused full left matrix implies 11,575,724,168 bytes versus
  912,936 bytes for the reduced left matrix. A debugger attachment was unavailable;
  no system permission or ptrace setting was changed.
- A separate cached evaluation changed **only** those three normal helpers'
  requests to `full_matrices=False`, using a process-local wrapper. It retained
  the original placement, solver/backend, thresholds, cloud and poses; it did
  not rerun the trajectory or change any repository source. Three synthetic
  face/plane comparisons against the original helpers agree to 1e-12. It
  completed in 201.99 seconds and preserved its exact placement for inspection.
  These timings are different workloads, not a measured speedup ratio.

`execution.json` records the unchanged-run timeout. The completed report is
`cached-evaluation.stdout.json`, with its own execution record and equivalence
preflight. These outcomes must not be conflated. The evaluator optimization is
diagnostic-only; it has not been promoted into frozen production/tool sources.

## Geometry and coverage

| Quantity | Current result | Historical note |
| --- | ---: | ---: |
| CAD-side observed-point RMS | 0.50161 mm | about 0.50 mm |
| CAD-side p95 | 0.89636 mm | about 0.89 mm |
| CAD reference points observed | 95.0417% | about 95% |
| Surfaces under historical 1.5 mm threshold | 23/24 judged | 23/24 |
| Worst face orthogonality deviation | 0.74911 degrees | about 0.75 degrees |
| Weakest image-matched revisit agreement within 1.5 mm | 98.2011% | not a dimension test |

The historical surface gate still returns **fail**, not pass: pocket wall RMS is
1.60201 mm with 67.99% observed. Pocket-floor coverage is only 10.91%, so its RMS
is not reported. The `unobserved_surfaces` convenience list is empty because its
cutoff is 5%; that does not mean the pocket was adequately scanned. Loop closure
is partly enforced by the fixed-axis/rate model and cannot independently prove
shape accuracy or scale.

## Explicit dimensions

| Dimension | Owner's calipers | Scan base-plane separation | Difference |
| --- | ---: | ---: | ---: |
| X / width | 91.900 mm | 91.01569 mm | -0.88431 mm (-0.9623%) |
| Y / depth | 81.850 mm | 81.06712 mm | -0.78288 mm (-0.9565%) |
| Total height | 55.850 mm | not measured: bottom absent | unknown |
| Pocket depth | 12.100 mm | not claimed: floor coverage inadequate | unknown |

Method: independently fit the two opposing base faces, allowing each face's
slopes and intercept to vary. Measure their separation at the same base
mid-height in the rigidly placed reference coordinate frame. Do not fit scale
to match the calipers. The 3 mm and 5 mm selection windows produce identical
retained points and dimensions. Each face retains 3,658-7,615 points with broad
spatial support; normal RMS is 0.130-0.166 mm. Separation varies over the supported
length from 90.956-91.075 mm in X and 80.986-81.148 mm in Y.

Six helper tests pass, including deliberate dimensional offsets, rigid-transform
recovery, rejection of scale disguised as rigid placement, a missing opposite
face, and rank-deficient points. Test evidence is `dimension-tests.xml`.
These checks validate the calculation, not the scanner's physical accuracy.

Base-face separation is not necessarily the maximum caliper extent of a print
with a flared first layer or local warping. The same fractional error in X/Y is
a useful clue, not proof of a uniform depth-scale fault. It could involve optics,
reconstruction or print/measurement details. No approximately 1% compensation
is authorized by this one result. Missing repeated readings and instrument
resolution still limit uncertainty claims, without invalidating the practical
comparison.

## CAD inspection and preservation

FreeCAD 1.1.3 created and saved the document. Its ZIP contents passed integrity
checking. The views display 117,174 points, every third point from a simple
expanded bounding-box crop containing 351,520 points. This is a display crop,
not a distance-to-CAD filter. The full 389,785-point cloud and rigid transform
are preserved separately. Grey CAD surfaces are expectations, not measurements
of hidden areas; the scan-only views avoid that ambiguity.

The task-created FreeCAD process did not exit on its scripted window-close
request. After verifying the saved document and images, only that exact
command-validated process was terminated with SIGTERM. No pre-existing CAD
process or document was touched.

The initial replay's post-check confirms all 195 existing capture files and
their modification times unchanged, along with all 88 recorded source/reference
files. No scanner acquisition, firmware, calibration, exposure, driver or
dependency change occurred. Observed host CPU temperature reached 75 degrees C
and later read 63; no thermal alarm was observed. Those are spot checks, not a
thermal-soak qualification, and are not scanner internal temperatures.
M27 remains suspended, unchanged.

## Next physical step

Update: the [fresh controlled repeat](component-repeat-results.md) has now run.
Its base dimensions differ by about 0.10 mm, and the owner supplied new
mid-height X/Y calipers 91.77/81.74 mm. The linked additive comparison preserves
the original readings and CAD scores reported above without changing either
cloud or fitting scale. The earlier physical-step proposal follows for context.

The subsequent [fresh-repeat readiness check](component-repeat-readiness.md)
shows ramp/pocket/steps geometry consistent with the measured target, but the
scene is stationary and the colour view clips its left side. Recenter the whole
target, start the turntable slowly and leave the scanner fixed, then repeat
the acquisition with recorded settings. Visual consistency alone does not
establish physical print identity; an unrelated print must not inherit this
target's caliper record.
Compare a fresh result against the same dimensions, and obtain a separate higher
view for the pocket if needed. Establish whether the approximately 1% shortfall
repeats before changing calibration or calling the workflow reliable for chassis
clearances. Critical interfaces still require independent measurements.
