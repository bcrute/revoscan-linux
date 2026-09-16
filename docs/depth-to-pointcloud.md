# Depth frame to point cloud — what the geometry path does, and what it does not establish

Historical record: its "units unknown" statements predate the August 24
[caliper validation](metric-validation.md). See the September 9
[semantics review](reliability-calibration-semantics.md) for the current distinction
between measured units, software conversion and unresolved optical accuracy.
The original record below is preserved rather than rewritten as later evidence.

Status: the code path described here is offline. It reads bytes that a capture
already wrote to disk, applies arithmetic, and writes a file. It opens no device
node and sends nothing to the scanner. Every number quoted below was produced by
running that code on real 512000-byte frames captured from this unit.

**No distance this pipeline produces has ever been compared against a physical
measurement.** Not a ruler, not a caliper, not a certified target, not the
vendor SDK's own output. The pipeline is arithmetic over recovered constants,
and arithmetic produces plausible-looking numbers from any inputs at all. If you
read only one section of this document, read the next one.

## Read this first: four things that are not established

| Claim a reader might make | Standing | Why |
| --- | --- | --- |
| "the coordinates are millimetres" (or metres, or anything) | **Not established** | The `0x918` scalar was read from the device and its reciprocal decoded exactly. That proves what the device returned, not what it measures. No physical unit has been attached to it by any evidence in this repository. |
| "the principal point is right" | **Not established** | The factory `cx` sits 277.56 px left of the 320 px frame centre — effectively on the frame's left edge. That is what the device returned; whether it describes a cropped sensor window, a different field order, or a genuinely eccentric optical centre is undecided. |
| "the distances are accurate" | **Not established** | Nothing in this repository has compared a converted distance against any physical reference. There is no measured error, no tolerance, and no pass/fail anywhere in the project's evidence. |
| "the lens is accounted for" | **Not established** | The deprojection is an ideal pinhole. This unit's sibling `Distort` calibration key has never been read, and no distortion is applied or inverted anywhere in the path. |

These four are not caveats bolted onto a working measurement system. They are
the current state of the evidence. The code is built so a consumer cannot lose
them: they travel in `DepthConversion.limitations`, then in
`CloudProvenance.all_limitations`, and finally into the PLY header as
`comment limitation:` lines, because a PLY has nowhere else to put them.

What *is* established is narrow and worth stating precisely:

| Fact | Evidence |
| --- | --- |
| The device returns 640x400 `'Y16 '` frames of exactly 512000 bytes | scanner-009, bounded V4L2 capture; nine such frames are available offline |
| The 40-byte factory `Intrinsics` record for 640x400 reads `fx = fy = 898.6829833984375`, `cx = 42.43747329711914`, `cy = 192.8326873779297`, fixed entries `0,0,0,0,1` | scanner-007, one live read; blob SHA-256 `fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf` |
| The `0x918` property reply carried marker `4` and denominator `200`, giving `float32(1.0f / 200) = 0.004999999888241291` (bits `0x3ba3d70a`) | scanner-008, one live read; reply SHA-256 `9873fa19…37dd4` |
| Those three inputs compose into a point cloud without error, deterministically, with every sample accounted for | the offline run in [Worked example](#worked-example) |

The gap between the two tables is the whole subject of this document.

## The pipeline

Four modules, in dependency order. The two ends were written not to know about
each other, and the seam between them is a module of its own.

| Module | Responsibility | Entry point |
| --- | --- | --- |
| `scanner3d.v4l2_capture` | Owns the one depth mode: `FOURCC = 'Y16 '`, 640x400, 2 bytes per sample, `EXPECTED_FRAME_BYTES = 512000`. Everything downstream imports those numbers rather than restating them. | `V4l2DepthFrameBackend` (the device path; not used by anything below) |
| `scanner3d.depth_metric` | Turns one raw frame plus a calibration into camera-frame points, and the accounting that says what happened to every sample. | `deproject_depth_frame(raw, *, intrinsics, depth_scale, unit) -> DepthConversion` |
| `scanner3d.cloud_export` | The seam. Translates one module's records into the other's, and carries every caveat across. | `export_depth_frame(...) -> CloudExport`, `write_conversion_ply(...)`, `cloud_provenance_for(...)` |
| `scanner3d.pointcloud` | Encodes and writes the PLY, header comments included. Computes no coordinate. | `write_ply(path, points, *, provenance, ply_format, max_points, max_output_bytes)` |

```
512000 bytes ('Y16 ' 640x400)
        |
        |  struct.unpack("<256000H")  — exact length or refusal, nothing partial
        v
256000 uint16 samples, row-major from the top-left pixel, 1280 bytes per row
        |
        |  d = raw * depth_scale
        |  x = (u - cx) * d / fx        <- deproject_depth_frame
        |  y = (v - cy) * d / fy
        |  z = d
        v
DepthConversion:  points + per-reason exclusions + checks + limitations
        |
        |  cloud_provenance_for()  — units, convention, calibration, caveats
        v                              <- cloud_export
CloudProvenance + (x, y, z) triples in raster order
        |
        |  build_header() then binary32 body   <- pointcloud
        v
one PLY, written whole or not at all
```

Two structural properties of that diagram matter more than the arithmetic in the
middle of it.

**Every refusal happens before anything is read or written.** A blank unit
assumption, unusable intrinsics, an unusable scale, a buffer that is not exactly
512000 bytes, a blank calibration source, a malformed intrinsics digest, a
conversion whose accounting does not add up, a non-finite coordinate, a file past
a ceiling — each is raised before a single sample is unpacked or a path is
opened. A refused export leaves no file behind, not even an empty one.

**The output is deterministic.** Points are built in the frame's own raster
order and exported in that order, so two exports of one conversion are
byte-identical files. That is what makes a future validation run comparable
against an earlier one.

## The arithmetic, exactly

One expression, written once in the code as `DEPROJECTION_EXPRESSION` so the
report and the implementation cannot disagree:

```
d = raw * depth_scale
x = (u - cx) * d / fx
y = (v - cy) * d / fy
z = d
```

where `u` is the column index (0..639), `v` the row index (0..399), and
`raw` the uint16 sample at `index = v * 640 + u`. Nothing else happens to a
sample. There is no temporal filtering, no hole filling, no smoothing, no
outlier rejection, no plane fitting and no distortion term.

The four inputs, and where each comes from:

| Symbol | Value on this unit | Source | Standing |
| --- | --- | --- | --- |
| `fx`, `fy` | `898.6829833984375` | factory `Intrinsics` record, scanner-007 | read from the device, in pixels of the 640x400 frame it names |
| `cx` | `42.43747329711914` | same record | read from the device; **277.56 px left of centre** |
| `cy` | `192.8326873779297` | same record | read from the device; 7.17 px above centre |
| `depth_scale` | `0.004999999888241291` | `0x918` property read, scanner-008 | the exact reciprocal the device returned; **physical unit unknown** |

No module in the geometry path carries any of those numbers as a default.
`deproject_depth_frame` takes an injected `IntrinsicsRecord`, an injected scale,
and — required, never defaulted — a `DepthUnitAssumption` naming what the caller
asserts `raw * depth_scale` measures, who is staking that, and on what basis.
The one such record the repository ships is `UNVALIDATED_UNIT`, whose basis
field reads, in full:

> none. The 0x918 scalar was read from the device and its reciprocal decoded
> exactly, which establishes what the device returned and not what it measures;
> no physical measurement has related a converted distance to millimetres,
> metres or anything else

Declining to name a unit is therefore a decision that appears in the report,
rather than a default that hides in it.

### The trap

The recovered scalar is bit-identical to `0.005f`, the constant the vendor SDK
falls back to for a MINI when it has no device answer. That agreement is an
observation and nothing more. Two numbers agreeing does not give either one a
unit, and the inference that the SDK's authors chose `0.005f` *because* it
matches this device is, at most, something the agreement is consistent with.

The related trap is arithmetic. Multiply the frame's codes by the scalar, look
at the results, and they will have some magnitude that feels reasonable for
something. That feeling is not evidence: for any constant factor there exists a
unit name that makes the products look sensible. A scale that is wrong by a
factor of ten, or by 25.4, or by 1.06, produces exactly the same impression of
reasonableness. Only a physical comparison separates those cases, and none has
been made.

## The coordinate convention

Spelled out axis by axis rather than named, because "camera frame" means at
least three different things across the graphics, robotics and computer-vision
worlds, and a cloud that is silently mirrored relative to what a reader assumed
is worse than no cloud.

| Element | This pipeline's convention |
| --- | --- |
| Origin | The camera's optical centre — the point that projects to the principal point `(cx, cy)`. `z = 0` there. |
| `+x` | Toward increasing column index `u` (image right). `x = 0` lies on the `cx` column, **not** on the frame's centre column. |
| `+y` | Toward increasing row index `v` (image down). `y = 0` lies on the `cy` row. |
| `+z` | Along the optical axis, away from the camera, into the scene. |
| Handedness | Right-handed: `x̂ × ŷ = ẑ`, with x right, y down, z forward. |

**This is the convention this code applies, not one the device declared.** It is
the convention implied by the arithmetic above and by nothing else. Nothing
measured on this unit establishes the handedness or the frame convention of the
factory record (`docs/architecture.md`, Domain 4). A consumer that expects y-up
or z-back must transform these points itself; it must not assume the transform
already happened.

That sentence is not decoration, and the code treats it as load-bearing.
`depth_metric.CoordinateConvention` carries it in a `note` field;
`pointcloud.CoordinateConvention` — a different record, with a `name` field the
first lacks and no `note` field at all — has nowhere to put it. A field-by-field
copy between the two would silently drop it and produce a PLY header that states
an axis convention as though the hardware had confirmed it.
`cloud_export.convention_caveat` exists to prevent exactly that: the note is
carried across as a *limitation*, which is the block a reader is directed to for
facts of that class, and it lands in the header as its own
`comment limitation:` line.

### What the off-centre principal point does to the cloud

`cx = 42.44` on a 640-wide frame puts the optical axis 2.7° inside the frame's
left edge. The resulting angular extents, computed from the record's own values:

| Edge | `(index - c) / f` | Angle from the optical axis |
| --- | --- | --- |
| column 0 | -0.047222 | -2.70° |
| column 639 | +0.663819 | +33.58° |
| row 0 | -0.214573 | -12.11° |
| row 399 | +0.229411 | +12.92° |

So the horizontal field runs from -2.70° to +33.58° — 36.28° wide and almost
entirely on one side of the axis — while the vertical field is nearly symmetric
at ±12.5°. Every cloud this pipeline produces from this record inherits that
asymmetry. It is visible directly in the run below, whose x extent is
`[-3.09, +72.55]` against a z extent of `[0.42, 254.57]`.

The consequence for any future validation is sharper than "the cloud looks
lopsided". If the true principal point were the frame centre, using `cx = 42.44`
adds `(320 - 42.44) · d / fx = 0.3089 · z` to every x coordinate. That is a
**shear along z, not a rigid shift** — and a shear maps planes to planes. A flat
target and a plane fit therefore cannot detect this error at all. Only a target
with known angles, or known extent measured across a range of depths, can.

This is surfaced, never corrected. Recentring a factory record to make a cloud
look symmetric would be inventing a calibration.

## Zero is not a distance

A raw sample of `0` is read by this repository as "the sensor returned no depth
for this pixel". Such samples are excluded, counted, and named — never converted
to a point.

The reason this matters is quantitative. Across the nine real frames captured
from this unit and available offline:

| Frame (SHA-256 prefix) | Zero samples | Points built | Yield | Smallest non-zero code |
| --- | --- | --- | --- | --- |
| `9fcf14e78a2d7bc7` | 249769 | 6231 | 2.43% | 81 |
| `793ab4fff21c5e1c` | 249756 | 6244 | 2.44% | 80 |
| `6b135e5193a78507` | 244958 | 11042 | 4.31% | 17 |
| `24aec75c13732739` | 243751 | 12249 | 4.78% | 21 |
| `97ade34e86c88b48` | 173574 | 82426 | 32.20% | 328 |
| `05b8065ef085c34b` | 170110 | 85890 | 33.55% | 85 |
| `c890d38b33c32a61` | 170044 | 85956 | 33.58% | 85 |
| `ad0aea913318848b` | 166930 | 89070 | 34.79% | 84 |
| `de2897d87cd9773d` | 162299 | 93701 | 36.60% | 25 |

Zeros are 79.5% of all 2304000 samples in that set, and up to 97.6% of a single
frame. If they were converted, `x = (u - cx) · 0 / fx = 0` for every one of
them, so they would all land at exactly `(0, 0, 0)` — and in the worst frame
above, 97.6% of the cloud would be a single point at the camera origin,
surrounding the 2.4% that actually describes a surface.

Two further observations support the "no return" reading, and neither is proof:

- **The low codes are empty.** Across all nine frames — 2304000 samples — exactly
  three samples carry a code below 50, and the smallest non-zero code seen
  anywhere is 17. If `0` were a legitimate short-distance measurement, the codes
  would be expected to run continuously down toward it rather than stopping
  dead.
- **Zeros are regions, not noise.** Between 81% and 99.8% of interior zero
  samples have all four of their orthogonal neighbours also zero, depending on
  the frame. That is the spatial signature of background and shadow, not of
  scattered per-pixel dropouts.

**The firmware's own invalid-sample convention is undocumented.** No other
sentinel value is recognised by this code. If the firmware also uses, say, a
saturated `0xFFFF` for "too far", this pipeline would convert it into a point at
327.675 assumed units and nothing here would notice. That is an open gap, not a
solved problem.

## The exclusion accounting: every one of the 256000 samples

`PIXELS_PER_FRAME = 640 × 400 = 256000` samples go in. Points built plus
per-reason exclusion counts add back up to exactly that, and
`DepthConversion.accounting_holds` says so as a property a caller can assert on.
`cloud_export` re-checks it and refuses to write a file from a conversion whose
accounting does not reconcile.

The partition is by *first applicable reason*, tested in the order below, so a
sample that would qualify for two is counted once and the total stays a
partition rather than a tally that can double-count. Every reason appears in
every result with `count = 0` when it did not occur, so a reader never has to
first discover which reasons a particular frame happened to trigger.

| Order | `ExclusionReason` | Meaning | Can it occur with this unit's inputs? |
| --- | --- | --- | --- |
| 1 | `no_return` | The raw sample was `0`. Not a measurement of zero distance. | **Yes — this is the only one that fires in practice.** 63%–98% of a real frame. |
| 2 | `non_finite_distance` | `raw * depth_scale` was not finite. | No. A uint16 times a finite positive scale is always finite. Counted rather than assumed away so that an extreme injected scale narrows a cloud instead of poisoning it. |
| 3 | `point_limit` | The fixed 256000-point ceiling had been reached. | No, structurally: the ceiling equals the sample count, so reaching it requires every sample to have already produced a point, by which time the frame is consumed. It exists so that the bound is enforced by construction rather than by argument. |
| 4 | `non_finite_coordinate` | `d` was finite but `(u - cx)·d/fx` or `(v - cy)·d/fy` overflowed. | No, with a scale of this magnitude. A point carrying an infinity would silently destroy any extent, centroid or fit computed downstream, so it is excluded and counted rather than written. |

On real data the partition therefore reduces to **points + zeros**, exactly.
That is the intended outcome, not a coincidence: the other three reasons exist so
that the accounting is *total* rather than conditional on the inputs being
reasonable, and so that a future caller who injects a hostile scale gets a
narrower cloud and a full count instead of a corrupted one.

Reading the accounting from a report:

```
pixel_accounting.samples            256000   = width * height, always
pixel_accounting.included           89070    = len(DepthConversion.points)
pixel_accounting.excluded           166930   = sum of every counts_by_reason entry
pixel_accounting.accounted          256000   = included + excluded
pixel_accounting.accounting_holds   true     = accounted == samples
```

A gap in the resulting cloud is an **absent measurement**, not observed empty
space. No surface, hole or edge may be inferred across one. That sentence is
written into every file's header by `cloud_export._survival_caveat`, together
with the per-reason counts, precisely because a hole in a point cloud renders
identically to a region that was measured and found empty.

## What reaches the file

A PLY header records types and nothing else. It has no field for a unit, no
field for axis directions, and no field for provenance, so a consumer handed a
bare PLY guesses — and for this scanner every available guess is wrong. The only
place those facts can live is `comment` lines, and `pointcloud.build_header`
writes them there:

| Header comment | Content |
| --- | --- |
| `produced by` | `scanner3d <version>` |
| `why these comments exist` | that the header is the only place units, axes and provenance can be stated |
| `units` | the caller's `DepthUnitAssumption`, prefixed `ASSUMED, never measured:`, with its `stated_by` and `basis` |
| `dimensional accuracy` | `none is claimed; see the limitations below` |
| `coordinate convention` | the one-line axis summary |
| `source frame sha256` | the digest of the exact frame that produced the points |
| `calibration` | source, intrinsics digest, the scale value, and the scale's unit disclaimer |
| `limitation` (one per) | `INHERENT_LIMITATIONS`, then the convention note, the survival caveat, every `DepthConversion` limitation, and every calibration observation |

In the run below that is 14 distinct limitations and a 6328-byte header on a
1075168-byte file — about 0.6% of the file spent saying what the other 99.4% does
not mean. `INHERENT_LIMITATIONS` is prepended by the record itself rather than
passed by the caller, so no call site can produce a cloud from this repository
that omits the uncertainty it inherits, and the caller's own limitations are
appended rather than substituted.

Two more guards worth knowing about, because they change what a file can be:

- Caller-supplied provenance strings are flattened to single lines of printable
  ASCII before they reach the header. A units label containing a newline and the
  token `end_header` would otherwise terminate the header early and turn the
  rest of the file into garbage.
- The header is refused, never truncated, past 8192 bytes. Half a stated caveat
  reads like a whole one.

## What would be required to validate any of this

None of the following has been done. It is written here as a specification of
absent work, so that a future run can be judged against something predeclared
rather than assembled after the numbers are in.

**1. A target of known dimension.** A rigid object whose relevant dimension is
known independently of this scanner — a machined gauge block, a printed
checkerboard whose pitch has been measured with calipers, a pair of parallel
planes at a measured separation. The dimension must be recorded *with its own
uncertainty* (the caliper's resolution, the printer's tolerance), because a
comparison against a reference of unstated precision cannot fail honestly.

**2. A tolerance declared before the run.** Written down before any frame is
captured: which quantity is compared, over how many poses, and what deviation
counts as a failure. A tolerance chosen after seeing the data is not a test.
Without this step the entire exercise collapses into finding a unit that makes
the numbers look right, which is the failure mode this pipeline is built to
resist.

**3. A protocol that separates the failure modes.** The measurement must be
capable of distinguishing at least these outcomes, which a single frame of a
single flat target at one distance cannot:

| Observation | What it would establish |
| --- | --- |
| ratio of measured to true dimension is constant across distances and ≈ 1 | the assumed unit label is the right one, to the declared tolerance |
| ratio is constant across distances but ≠ 1 | the model is right and the scale is wrong by that factor; the unit follows from it |
| ratio varies systematically with distance | the model is wrong, not merely the unit — a linear `raw × scale` is not the device's mapping |
| ratio varies across the frame at fixed distance | lens distortion, or the principal point, or both — and the two are separable only by moving the target |

That last row is why the target must be placed at several positions across the
frame, not only near its centre, and why the shear consequence of `cx` described
earlier matters: a centred flat target with a plane fit is exactly the
experiment that would *pass* while `cx` is wrong.

**4. A recorded, reproducible capture.** Frame digests, the intrinsics digest,
the scale, the exposure and illumination state, and the exact code revision, so
that a disagreement later can be traced to an input rather than argued about.
The pipeline already emits the first three in every report; the rest is the
protocol's job.

**5. A statement of what is still not established afterwards.** Even a clean
pass would establish the unit and the accuracy *of that dimension, at those
distances, in that part of the frame, on this one unit, at that temperature*. It
would not establish distortion behaviour at the frame edges unless the target
went there, would say nothing about RGB alignment, and would not transfer to
another MINI whose factory record differs.

Until items 1–4 exist, `UNVALIDATED_UNIT` is the correct thing to pass, and any
unit label a caller supplies instead is that caller's assertion — recorded as
theirs, beside the word *assumed*, in every report and every file header.

## Worked example

Real, and run offline on 2026-08-24 against a real captured frame. The
intrinsics are not typed as floats but parsed from the 40 bytes scanner-007 read
off the device, whose SHA-256 is quoted above and which a reader can verify:
`sha256(BLOB)` is `fd0583b9…0011bf` exactly.

```python
import pathlib

from scanner3d.cloud_export import export_depth_frame
from scanner3d.depth_metric import UNVALIDATED_UNIT
from scanner3d.intrinsics import parse_intrinsics
from scanner3d.pointcloud import PlyFormat

# The 40-byte factory Intrinsics record read from /data/camparam/Pl.bin
# (scanner-007). Parsed, never hand-assembled into a record.
BLOB = bytes.fromhex(
    "80029001b6ab604400000000f9bf294200000000b6ab60442bd5404300000000000000000000803f"
)
SCALE = 0.004999999888241291  # float32(1.0f / 200); denominator 200, marker 4

record = parse_intrinsics(BLOB)
assert record is not None

export = export_depth_frame(
    pathlib.Path("cloud.ply"),
    pathlib.Path("depth-frame-raw.bin").read_bytes(),
    intrinsics=record,
    depth_scale=SCALE,
    # Declining to name a unit, out loud, in the file.
    unit=UNVALIDATED_UNIT,
    calibration_source=(
        "scanner-007 depth-intrinsics read of this unit; scanner-008 depth-scale read"
    ),
    intrinsics_sha256=("fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf"),
    ply_format=PlyFormat.BINARY_LITTLE_ENDIAN,
)
```

What that call returned, for the frame with digest `ad0aea91…5642f`:

```
samples            256000
points written     89070
every point kept   True
accounting holds   True
counts by reason   {'no_return': 166930, 'non_finite_distance': 0,
                    'point_limit': 0, 'non_finite_coordinate': 0}
file               cloud.ply   1075168 bytes  (6328-byte header + 89070 x 12)
cloud sha256       e7785ba263a3043e95ff70d756447d0d5ebb29a35fc5f04f5c9a439f62b98eee
limitations        14
truncated          False
checks             intrinsics_frame_matches_depth_frame  True
                   principal_point_near_frame_centre     False
                   intrinsics_structurally_expected      True
extent  x [-3.0906, 72.5502]   y [-42.6422, 27.4627]   z [0.4200, 254.5650]
        raw [84, 50913]
first point  u=4 v=0 raw=328  x=-0.070144 y=-0.351899 z=1.640000
```

The extent and first-point figures are rounded for display here; the record and
the file carry the full values the code produced.

Read that output the way it is meant to be read. `89070` points came from
`256000` samples; the other `166930` are `no_return` and are absences, not empty
space. The x extent is almost entirely positive — -3.09 against +72.55 — because
x is measured from a principal point 278 px left of the frame centre, which is
also why `principal_point_near_frame_centre` is `False`: an observation the code
reports and refuses to act on. The x, y and z figures are in the caller's assumed
unit, and this caller explicitly declined to name one; `raw [84, 50913]` is in
uint16 sensor codes and is the only extent in that block whose unit is known.

`CloudExport` carries both halves of the evidence: `.conversion` (the full
accounting), `.cloud` (the file record), plus `.path`, `.point_count`,
`.header`, `.limitations`, `.every_point_written`, and `.as_dict()` for a
`schema_version: 1` report to serialise beside the PLY.

For a cloud small enough to read by eye, pass
`ply_format=PlyFormat.ASCII`; the ASCII body writes the `repr` of the same
binary32-rounded values the binary body packs, so the two encodings of one cloud
decode to identical coordinates.

### The same conversion from the command line

`scanner3d depth-frame-to-ply` is the `export_depth_frame` call above with the
calibration typed rather than imported — the same three modules, the same
refusals, the same header. It is a **file-to-file transform**: it resolves no
scanner, opens no device node and runs no vendor-SDK probe, so it works with the
hardware unplugged and on a frame captured months ago. The invocation below was
run offline on 2026-08-24 against the same frame:

```
$ scanner3d depth-frame-to-ply base_4/depth-frame-raw.bin cloud.ply \
      --fx 898.6829833984375 --fy 898.6829833984375 \
      --cx 42.43747329711914 --cy 192.8326873779297 \
      --depth-scale 0.004999999888241291 \
      --calibration-source "scanner-007 depth-intrinsics read of this unit; scanner-008 depth-scale read" \
      --intrinsics-sha256 fd0583b92b0e16e6f4d708d5ab64e57981de9ba92f6c9587b6aae3b8f20011bf
```

It wrote the byte-identical file the library call above did: 1075168 bytes,
89070 vertices, SHA-256
`e7785ba263a3043e95ff70d756447d0d5ebb29a35fc5f04f5c9a439f62b98eee`. Abridged,
its output reads:

```
scanner3d depth-frame-to-ply 0.1.0
THESE COORDINATES ARE NOT METRICALLY VALIDATED. ...

Source frame (read from disk; no device was opened, no SDK probe was run)
  bytes      512000 (a 'Y16 ' 640x400 frame is exactly 512000)
  sha256     ad0aea913318848b5bf82911ff28c7d4b7c8386934f6b3e771b3ef9edfc5642f

Calibration (yours: this command reads none from any device)
  fx 898.6829833984375  fy 898.6829833984375  cx 42.43747329711914  cy 192.8326873779297
  scale      0.004999999888241291
  applied as d = raw * depth_scale; x = (u - cx) * d / fx; y = (v - cy) * d / fy; z = d
  SCALE UNIT unknown from this evidence: ...

Assumed unit (an assertion, never a measurement)
  label      unvalidated depth unit
  stated by  no one: this module declines to name a unit
  basis      none. The 0x918 scalar was read from the device ...

Conversion
  points     89070 of 256000 samples became points
  excluded   166930 produced none (no_return=166930, non_finite_distance=0,
             point_limit=0, non_finite_coordinate=0)
  accounted  256000 of 256000 (every sample accounted for: True)
  extent     x [-3.0905877159058006, 72.55018186104314]  ...
             raw codes [84, 50913]; the x, y and z spans are in the assumed
             unit above and are not measured lengths
  observations (made and deliberately not acted on)
    [ok  ] intrinsics_frame_matches_depth_frame: ...
    [note] principal_point_near_frame_centre: x = 0 lies on column 42.4375 ...
    [ok  ] intrinsics_structurally_expected: ...

Point cloud
  vertices   89070 (every converted point written: True)
  bytes      1075168
  sha256     e7785ba263a3043e95ff70d756447d0d5ebb29a35fc5f04f5c9a439f62b98eee

LIMITATIONS (every one of these is also written into the PLY header)
  - ...fourteen lines, identical to the header's...

THESE COORDINATES ARE NOT METRICALLY VALIDATED. ...
Status: wrote 89070 vertices to cloud.ply
```

`--fx`, `--fy`, `--cx`, `--cy`, `--depth-scale` and `--calibration-source` are
all **required**, and that is not a style choice: they are the arguments
`export_depth_frame` refuses to default, because the caller alone can know which
calibration is being applied and where it came from. `--assume-unit` is
optional, and omitting it is the honest default — the file then records that a
unit was *declined* rather than naming one; supplying one records it as the
caller's own assertion, with the caller named as its author.
`--intrinsics-width` / `--intrinsics-height` default to the frame's own 640x400,
and a record describing another raster is converted as asked with the mismatch
reported, never rescaled. `--ply-format`, `--json` and `--json-out` behave as
they do elsewhere in the CLI.

There is deliberately **no `--calibration-file`, no `--intrinsics-json`, and no
flag that reads a value off the device** — not an omission to be filled in later.
A stored profile or a silent factory read would let this unit's numbers reach a
cloud without anyone having asserted them, which is the precise failure this
document exists to prevent: the resulting file would carry a calibration whose
author is nobody. Typing them is the point.

Four exit codes, all distinct, so a script can tell a refusal from a failure:

| Code | Meaning | When |
|---|---|---|
| `0` | The file was written | Including a zero-vertex cloud — see below |
| `3` | `PERMISSION_DENIED` | The output name already exists; this command never replaces a file |
| `5` | `UNSUPPORTED_MODE` | A wrong-sized frame, and every refused input: `fx`/`fy` <= 0, a non-finite or non-positive scale, a blank `--calibration-source`, a blank `--assume-unit`, a malformed `--intrinsics-sha256` |
| `7` | `INTERNAL_ERROR` | The frame path is unreadable or not a regular file |

Every refusal prints to stderr, writes nothing to disk, and emits no JSON even
under `--json`.

**A zero-vertex cloud is exit `0`, not an error.** A frame whose every sample is
`0` is a frame on which the projector got nothing back — a target too dark, too
far, too glossy or absent — and that is an outcome of the scene, not a fault in
the conversion. The command writes the file and the accounting still balances at
256000, all of it `no_return`; the `Point cloud` block then carries an `EMPTY`
row that makes the same distinction [absence is not
emptiness](#zero-is-not-a-distance) draws above:

```
  EMPTY      this file has zero vertices: the conversion ran and found no sample
             with depth. That is a record of a run, not of an empty space
```

Treating that as a failure exit would be the tool claiming to know that the user
pointed it at the wrong thing, when all it knows is that nothing came back.

The ordering of that output is deliberate. The not-validated banner is printed
**before the first number appears**, so a reader who stops at the vertex count
has already been told the count is not evidence of a measured shape; the assumed
unit is labelled as an assertion where it appears; the extent block says in
place that its x, y and z spans are not measured lengths while `raw codes` is
the one row whose unit is known; and the full limitation block — the same
fourteen lines the PLY header carries — comes last, with the banner repeated
after it and only the one-line status after that. Run
`scanner3d depth-frame-to-ply --help` for the current surface, and treat any
disagreement with this section as this document being out of date.

## See also

- `src/scanner3d/depth_metric.py` — the deprojection and the accounting
- `src/scanner3d/cloud_export.py` — the seam, and why the convention note is carried as a limitation
- `src/scanner3d/pointcloud.py` — the PLY encoder and its bounds
- `README.md` — the measured `depth-intrinsics` and `depth-scale` runs, in full
- [`docs/architecture.md`](architecture.md#domain-4--calibration--depth-scale-fact-disassembled) — Domain 4, where the depth scale and the open frame-convention question are recorded
