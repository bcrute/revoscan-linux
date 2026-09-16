# Existing staircase: second, more downward view — 2026-09-10

## Outcome and next decision

This view supplies wider image footprints of both measured tread tops. The
preselected primary spacing is **9.931477 mm** at 10 ms versus the provisional
physical difference **9.90 mm**, compared with 10.243219 mm in the first view.
That is an improvement in agreement, **not a passed accuracy or calibration
gate**. The position, distance and sampled physical patches also changed, so
the comparison does not isolate camera angle alone.

Within this new stopped view, 10 ms improves surface shape and reduces fitted
nonparallelism. A separately declared common-pixel control confirms that this
is not explained only by a change in missing-pixel selection. The remaining
patch/evaluation dependence and unknown physical-reference uncertainty prevent
choosing one scalar height as a transferable calibration reference.

Keep the present scanner angle; no additional print or immediate reposition is
requested. Next: repeat the **underside-to-top** heights of these same two treads
at identified positions along the tread (near front, centre, near back, avoiding
the rounded edges), keeping the separate readings. Record caliper model or
stated accuracy and display resolution if available. This specifically probes
physical height variation along the stair direction, rather than assuming the
two printed tops are perfectly parallel. It does not assume repeated readings
remove an instrument's systematic error.

Evidence root: `/home/ben/scans/diagnostics/20260910-staircase-downward-iRDqsk/`.
The [first-view evidence](component-staircase-results.md) remains unchanged.
See the two capture/analysis plans, fixed `regions.json`, raw first/last views,
`step-results.json`, 120 per-frame records, `COMMON-SUPPORT-PLAN.md`,
`common-support-results.json`, and both audits.
Sealed **17:35:03 UTC**, **407 files / 76,784,405 bytes** excluding the manifest;
manifest SHA-256 `796a3df3b48e79c8a8416606bf6d6312b4cc8f046a24dd860a496c7bf30f67e4`.
Document snapshots precede this seal metadata. All six input/source bindings,
the exact first-view sealed root and all 449 frozen live/saved source records
verify unchanged; M27 remains stopped. Do not modify the sealed root.

## Reference and clean acquisition

The [physical reference record](../reference/targets/reconstruction_v1/prints/physical-print-20260831-steps-20260910.json)
still contains one owner reading each: **16.08 and 25.98 mm from the underside**,
for the two highest staircase treads. Their provisional difference is 9.90 mm.
There is no new measurement of base thickness or the pocket plateau. Original
CAD adjustment, calipers and all earlier results remain unchanged.

Initial state at **17:22:57 UTC** verifies exact serial A22704522A6F00A64,
frame/exposure 20/20 ms, manual 0, gain 24 and CPU/NPU 57.6/56.9 C. The
unchanged guarded driver takes **20/10/20 ms exposure**, frame period 20 ms and
gain 24 throughout, with 40 retained frames after each settling boundary.

- **160 full deliveries / 120 settled raw frames**, zero sequence drops/shorts.
- Observed cadence **4.52635 fps**; this is not a throughput experiment.
- CPU/NPU maxima **58.9/58.2 C**, including final readback; not projector
  temperature or a thermal-equilibrium result.
- Original settings/gain verified after guarded quiet shutdown, no transport
  fault, stable post-close identity. No manual replug/reset or firmware write.
- First/last images preserve the same layout. Within-stage first-versus-later
  maximum p90 absolute depth changes are 0.265/0.065/0.225 mm. Baseline stripes
  remain visible; this statistic is not pure motion or independent pose truth.

The first offline render command was started before acquisition had written
`report.json` and failed with FileNotFoundError, before producing images or
accessing hardware. After confirmed acquisition completion, unchanged rendering
succeeded. No hardware acquisition was repeated.

## Fixed regions and results

Camera-depth profiles identify the tops versus risers. The following image
interiors were fixed before calculating any heights (x0,y0,x1,y1 exclusive):

| Region | Bounds | Candidate pixels |
| --- | --- | ---: |
| Upper primary | 295,63,360,78 | 975 |
| Upper shifted | 310,67,373,82 | 945 |
| Lower primary | 295,121,358,133 | 756 |
| Lower shifted | 307,125,370,137 | 756 |

This is post-capture visual selection, not a blinded accuracy test or matching
caliper contact locations. All four combinations pass the unchanged local
support/plane-fit checks in all 120 frames: **480 pairs, no fit refusals**.
At 10 ms all selected pixels return in every frame. Lower-primary mean return
fractions at the three stages are **95.9226%, 100%, 96.2632%**. All missing and
reserved-code handling remains explicit; no temporal average or imputation.

Unchanged factory intrinsics/depth scale and independent free planes are used.
No nominal geometry, caliper, forced parallelism, tracking, CAD pose or scale
enters a fit. Primary forward evaluation uses the upper normal at the lower
centroid; reverse uses the lower normal at the upper centroid. With nonparallel
planes these are different geometric quantities, not answers to average.

| Exposure stage | Primary forward median (mm) | Reverse median (mm) | Free plane angle median |
| --- | ---: | ---: | ---: |
| Initial 20 ms | 9.931875 | 10.374336 | 3.015 degrees |
| 10 ms | 9.931477 | 10.048318 | 0.789 degrees |
| Final 20 ms | 9.933749 | 10.422897 | 3.344 degrees |

At 10 ms, the primary forward values range **9.921429–9.940762 mm**. Across
the four fixed combinations, forward medians span **9.919151–9.964656 mm**,
reverse medians **9.965691–10.048318 mm**, and median free angles
**0.067–0.789 degrees**. Preserve all combinations; the most parallel pair or
the closest value to 9.90 mm is not promoted to replace the primary result.
Primary fitted separations over supported lower footprints span
9.88435–9.97749 mm across frames/footprints, not a calibrated error interval.

The primary point estimate is +0.031477 mm (+0.3179%) from the provisional
reference, just outside a 0.0297 mm / 0.3% limit. Unknown caliper contacts,
repeatability and accuracy, plus the patch/direction dependence, make an
accuracy decision **inconclusive**, not a near-boundary pass. This one small
height does not qualify lateral dimensions or the full chassis/CAD workflow.

## Fixed-pixel control: real exposure sensitivity within this view

After the primary result, fix each primary ROI to pixels returning valid codes
in **all 120 frames**, without selecting by residual or height: 975 upper and
701 lower pixels. The same lists are used in every frame. All 120 common-support
pairs fit; none replaces a primary result.

| Exposure | Forward median (mm) | Reverse median (mm) | Angle median | Lower all-point RMS median (mm) |
| --- | ---: | ---: | ---: | ---: |
| Initial 20 ms | 9.929581 | 10.360114 | 2.948 degrees | 0.242680 |
| 10 ms | 9.931007 | 10.053926 | 0.816 degrees | 0.027174 |
| Final 20 ms | 9.930800 | 10.382124 | 3.096 degrees | 0.242616 |

Lower retained-point RMS also changes about 0.211/0.0268/0.212 mm, so this
effect is not solely a few trimmed gross outliers. Upper all-selected RMS is
0.796/0.0223/0.926 mm, while upper retained RMS is 0.0334/0.0215/0.0333 mm;
there the gross-tail contribution must not be hidden by quoting only inlier RMS.
The changed missing-pixel mask alone cannot explain the improvement on this
fixed subset. This does not identify a decoder/saturation mechanism or establish
absolute correctness, a globally suitable 10 ms preset, or transferable scale.

## Verification and preservation

The same **20 preflight/lifecycle/geometry tests** pass before capture; the
same **7 reporting tests** pass afterward (27 existing tests rerun, not 27 new
definitions). Primary computation takes 1.16023 seconds; common-support control
0.31494 seconds. The primary audit checks 133 bound inputs, 120 unique raw
frames, 480 patch fits / pair results and **409,475 ordered selected valid pixel
samples including overlaps**. The common-support audit independently rebuilds
the raw masks and checks all 120 unchanged-helper results and stage summaries.
These are reporting/integrity checks, not independent metrology truth.

There is no production/default, calibration, firmware, mesh, tracking-method or
suspended M27 change. No acceptance criterion is promoted. Next is the physical
reference check described above, not another exposure/ROI search on this view.
