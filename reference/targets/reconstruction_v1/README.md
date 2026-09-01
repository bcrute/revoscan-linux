# Reconstruction target v1

This is the physical reference object for the scanner work that cannot be
settled by synthetic fixtures or by looking at a mesh. It is deliberately not
a decorative calibration cube. One controlled relative-motion scan exercises
the failure modes we need to separate: tracking, missing surface, unsupported
fill, edge loss, scale error, and the lumpy surface seen beside cleaner points.

Print [`reconstruction-target-v1.stl`](reconstruction-target-v1.stl). Import it
as **millimetres**, at **100% scale**. Before slicing, the displayed bounding
box must be exactly **92 x 82 x 56 mm**.

## What is on it

Viewed from above, `+y` is the rear and `+x` is the right:

```text
                         rear (+y)
        -x / left                            +x / right
    +-------------------------------------------------------+
    |  22 mm plateau     |  ramp: high end  |  20 mm step  |
    |  +-------------+   |        /         |  10 mm step  |
    |  | 9 x 10 mm   |   |       /          |   5 mm step  |
    |  | deep pocket |   |      /           |   2 mm step  |
    |  +-------------+   |     /            |  6 mm rail   |
    |                     |    /             |              |
    |  known-flat base    | low end          | smooth bump  |
    |  + 1.5/3 mm rails   |                  |              |
    +-------------------------------------------------------+
                         front (-y)
```

| Feature | What it distinguishes |
| --- | --- |
| 26 x 17 mm base plane and 23 x 21 mm plateau plane | local ripple/lumpiness without averaging it over the whole object |
| 20 x 66 mm inclined plane | planar error at a second incidence angle; also a strong asymmetric tracking feature |
| 12 mm deep open pocket | genuinely missing concave surface versus software filling unsupported space |
| Smooth two-axis 13 mm bump | blobby bias and ripple on curvature, not only on axis-aligned planes |
| Steps 2, 5, 10 and 20 mm above the base | edge position and dimensional preservation at several scales |
| Rails 1.5, 3 and 6 mm wide | thin-feature survival at about 2.3, 4.7 and 9.4 of the appliance's expected 0.64 mm TSDF voxels |
| Offset, unequal layout | one geometric answer during camera motion instead of a rotationally symmetric target |

The exact planes, landmarks, formulas and dimensions are in
[`reconstruction-target-v1.json`](reconstruction-target-v1.json). The bound
[`reconstruction-target-v1-reference.ply`](reconstruction-target-v1-reference.ply)
contains 12,000 area-weighted CAD surface points with normals and surface IDs.
It excludes only the bottom face that is against the print bed and cannot be
seen while the target remains on the turntable.

## Print it

Recommended FDM settings:

- matte medium-grey PLA or PETG; avoid transparent, glossy, very dark, or
  fluorescent material;
- 0.4 mm nozzle, 0.12-0.16 mm layers, at least three perimeters and five
  top/bottom layers;
- 15-25% infill is sufficient; this is not a load-bearing part;
- use the orientation already in the STL, with the 92 x 82 mm face on the bed;
- no supports and no model scaling;
- put the seam on the outer front (`-y`) wall if the slicer permits it, away
  from the measured planes, ramp, pocket, rails and bump.

Do not sand, fill, vapour-smooth, paint, or dimensionally compensate this first
print. Remove only loose strings and any brim outside the base. Those treatments
would make the physical object differ from its recorded manufacturing process.

The earlier physical MINI captures showed that a matte light target can
saturate at close range with the current fixed exposure. Medium grey gives us
more margin. If the only available material is white, keep the scanner near the
far end of the protocol below and watch the return-health readout rather than
assuming white is automatically better.

## Measure the individual print

The STL is exact digital truth; a printed part is not. Before its scan becomes
a physical gate, record the printer, material, nozzle, layer height, and the
caliper's resolution. Take at least three readings of each item below and keep
the individual readings, not only their mean:

| Quantity | Nominal |
| --- | ---: |
| overall left-right width | 92.0 mm |
| overall front-rear depth | 82.0 mm |
| bottom-to-ramp maximum height | 56.0 mm |
| bottom-to-plateau height | 22.0 mm |
| pocket opening, left-right | 9.0 mm |
| pocket opening, front-rear | 10.0 mm |
| plateau top to pocket floor | 12.0 mm |

Also take one clear photo from above and one from the side. Send the readings,
caliper resolution, print settings and photos as ordinary notes; you do not
need to edit the JSON or choose coordinate frames, regions, alignments, or gate
thresholds. Those are repository work.

The first print's owner-reported readings are preserved in
[`prints/physical-print-20260831.json`](prints/physical-print-20260831.json).
That record is ready for a first comparative scan and remains explicitly
provisional until its caliper resolution and print metadata are supplied.

## Scan it

The preferred first run uses a fixed scanner and a rotating target. For a rigid
object, that produces the same sequence of relative object-camera poses as an
orbiting camera, while removing hand motion and making the revisit repeatable.

1. Centre and secure the target on a turntable. Use a matte, preferably dark
   turntable surface and background so stationary surroundings do not become
   the easiest geometry for registration to match.
2. Fix the camera around **220 mm standoff**, looking downward about **35-45
   degrees**. Keep the whole target inside the valid depth footprint and confirm
   that the pocket floor is visible. Roughly 200-240 mm is the useful band for
   this 92 mm part under the current fixed exposure.
3. Start the scan, then rotate continuously and slowly for **two complete
   revolutions**. Around 60-120 seconds per revolution is appropriate; smooth
   motion and overlap matter more than exact speed.
4. Stop only after the second return to the starting view. The repeated views
   give the pose graph closure evidence throughout the sequence, not merely at
   one final frame.
5. Keep the camera fixed and do not lift, flip, rescale, recenter or otherwise
   handle the target during the capture. Never move the camera and target in
   the same evaluated run.
6. Preserve the raw capture and provide its directory. Do not export only a
   mesh; the raw frames are needed to compare algorithms on identical input.

A separate camera-moving scan remains useful later because the mapper appliance
will move its camera, but it is not required for this first reconstruction
quality gate. The relative geometry is the same; cable motion, operator motion
and background behaviour are not, so those operational effects stay a separate
test rather than being conflated with mesh quality.

If return health collapses, back up or reduce the face-on exposure before
continuing. A frame with no returns is evidence about the capture, not a hole
that completion is entitled to fill.

## What this can and cannot prove

The target gives us dense nominal geometry, predeclared planar regions,
distinct landmarks, known feature sizes, and a support-free controlled scan.
After the actual print is measured, it can quantify surface coverage,
unsupported geometry, dimensions, edge survival and lumpiness much more finely
than visual inspection.

It does not independently measure the camera's six-degree-of-freedom pose.
The shape makes wrong tracking observable through CAD disagreement and loop
closure, but exact per-frame pose truth would require an external tracker or a
calibrated motion rig. It also does not make the nominal STL tighter truth than
the printer: physical gates must include the measured print error and layer
surface in their uncertainty.

## Regenerate and verify

The generator uses only the Python standard library:

```bash
.venv/bin/python reference/targets/reconstruction_v1/generate.py
```

It rewrites the STL, reference PLY and manifest deterministically. The test
suite regenerates them in a temporary directory, requires byte-for-byte
identity, checks every digest, and independently proves that the STL is one
consistently oriented watertight 2-manifold with the declared bounds.
