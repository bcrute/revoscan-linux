# Calibration semantics review, 2026-09-09

Subsequent result, 2026-09-10: [historical can replay](can-replay-results.md)
recovers the original invocation and reproduces every retained per-frame
diameter. It also finds a dated Git binding for the separately rejected frame
and resolves historical aggregate-count discrepancies. The September 9 findings
below remain the record of what was established at that earlier checkpoint.

Read-only evidence root:
`/home/ben/scans/diagnostics/20260909-calibration-semantics-review/`.
No scanner access, calibration change, SDK binary download/execution, licence
bypass, driver/dependency installation or production modification.

## Chronology matters

The early [depth conversion record](depth-to-pointcloud.md) says units were
unknown. The later [caliper comparison](metric-validation.md), dated August 24,
establishes millimetre-scale output for its measured 58.0 mm can. It does not
establish current printed-target identity, accuracy across arbitrary views,
an independently measured standoff, or a complete distortion model.

The metric fixture manifest retains an early principal-point interpretation in
`what_this_establishes`. Its later `cause_analysis` and `spray_ab_at_242` fields
weaken that interpretation: the distance/surface signatures do not support
treating one residual indicator as a measured principal-point correction.
Preserve those chronological fields, rather than selecting the most convenient
sentence. The exact device-returned matrix is not permission to recenter it,
nor proof of arbitrary-view physical calibration.

All fifteen manifest-listed raw fixtures are copied and match their historical
SHA-256 values. The bounded local preservation contains 27 files / 8,182,227
bytes. The named rejected `can-sprayed-125-1.bin` also exists, but is absent
from that fifteen-frame list. Its separate 512,000-byte copy has SHA-256
`aba16a8974c56a4a48a899508c74f0d8674a71b8a449ec919f4b36b06b83c24b`.
That preserves its current bytes, not an unavailable historical hash binding
to the old rejected measurement. The original fifteen-frame numerical roster
is not silently expanded or replaced.

## Published conversion, not an executed SDK oracle

The manufacturer's public source is inspected at the repository's already
pinned commit `516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7`. The Z16 point-cloud
path passes depth scale and intrinsics to the header's conversion. That routine
sets Z to scaled depth and applies the same pinhole X/Y expressions as our code.
It does not normalize radial range or apply a distortion polynomial. Its
resolution rescaling is exactly one for the actual matching 640x400 record and
frames. This supports the software convention, not exact-unit optical accuracy.
[Header implementation](https://github.com/Revopoint/3DViewer/blob/516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7/thirdparty/3DCamera/include/hpp/Processing.hpp),
[viewer call site](https://github.com/Revopoint/3DViewer/blob/516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7/src/cscamera/process/pointcloudprocessstrategy.cpp).

Before conversion, the viewer optionally applies hole filling or spatial/
temporal smoothing. These are distinct processing options, not a calibration
correction. Nothing here enables them or infers their use in Revo Scan.
[Depth-processing source](https://github.com/Revopoint/3DViewer/blob/516ee9b64ed9a36f7db2b1303d4c9364e39ac8d7/src/cscamera/process/depthprocessstrategy.cpp).

The public display-normal routine uses image-neighbor triangles rather than
our sampled-cloud PCA. That is not evidence about the proprietary application's
registration normals. A suspicion about an uninitialized local normal is
disproved by the same header's zero-initializing `float3` constructor; no such
bug is claimed. Shared accumulation in OpenMP loops remains an inspection
question, not an executed failure or an explanation of this unit's tracking.
The header and its original licence remain
in the external evidence archive, not installed into the application.

Eight selected public text files, their retrieval records and the bounded
730,853-byte Git tree response are preserved. The SDK's top-level header hash
matches the existing repository pin. The old `/tmp` SDK directory is absent,
the SDK-root environment setting is unset, and bounded accessible searches
find no SDK. Some system directories were inaccessible; this is not a complete
machine-wide absence claim. An initial incorrect header URL returned 404;
the actual `include/hpp/Types.hpp` path was then read and preserved.

## Numerical comparison

A separately declared test processes every nonzero pixel in all fifteen
preserved metric frames: **1,038,206 points**. The repository's actual scalar
deprojection matches an independent float64 expression exactly. Explicitly
rounding each published arithmetic operation to float32 changes a coordinate
vector by at most **0.000019361804013928656 nominal mm**. This is a NumPy
transcription, not execution of the SDK or compiler/FMA parity certification.

That direct difference is much smaller than the observed registration errors.
It does not prove iterative tracking is insensitive to tiny perturbations:
prior keyframe/history tests demonstrate otherwise. No correction is fitted.
The first script invocation failed before processing because it used absent
record attributes; its exact source is retained and the correction uses the
existing checked accessor. Both the fixed protocol and complete numerical
report are retained. The historical can measurement was not remeasured.

The optional-filter observation motivates the separately declared M26 derived-
depth experiment. Its unchanged raw geometry guards and survivor qualification
remain binding. Current-target identification and independent physical geometry
still require their own deliberately controlled evidence.

The later [device-file review](firmware-device-review-20260909.md) adds fresh
binary/YAML numerical comparisons and preserves the complete observed
calibration directory. Its exact rectified-matrix agreement is not a new
caliper experiment, authenticated metadata or permission to change calibration.
