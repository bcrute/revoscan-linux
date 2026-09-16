# Read-only organized-normal feasibility review

Declared September 9, 2026, 07:39 UTC. Evidence:
`/home/ben/scans/diagnostics/20260909-organized-normal-review/`.
This is not a tracking mechanism or a new qualification result. It can run
CPU-only alongside M26 without a third GPU workload or scanner access.

Existing experiments change sampled-cloud PCA, its support size or weighting,
and native-frame versus pooled-model provenance. This diagnostic instead
compares a direction computed directly from neighboring original image pixels.
The [official PCL tutorial](https://pointclouds.org/documentation/tutorials/normal_estimation_using_integral_images.html)
describes image-space gradient and covariance approaches; this immediate,
unsmoothed 3x3 cross product is not its averaged implementation or evidence of
the proprietary scanner software's tracking method. No dependency is installed.

Use the complete earlier 192-cloud raw roster, including both capacities.
Verify original prepared/retained/voxel points and both saved PCA fields exactly.
Each unchanged voxel centroid uses the nearest original contributor, with
original-index ties, solely to identify a measured pixel. Its exact original
deprojection must reproduce the point. Horizontal and vertical neighbor XYZ
differences define the diagnostic direction. All nine samples must be inside
the original band with <=1.5 nominal-mm code span; borders, holes, depth edges
and degenerate cases remain explicitly unavailable. No PCA fallback or fill.

Retain every direction, status, source pixel/contributor, tangent lengths and
sign-invariant angle to each original field. Neither field is physical truth;
a representative pixel is not proof of the true voxel/model normal. No fitted
pose, object/support mask, geometry gate, raw byte or M26 source is changed.

Initial 64 focused checks pass after a retained import-path collection failure;
79 expanded checks also pass. Broad selected verification passes 1,544 tests,
clean lint and cold typing of 131 files. Executing helpers and selected-test
sources are preserved before the 192-cloud review starts at 07:47 UTC.
No conclusion about normal accuracy or a new tracking substitution follows
from these implementation checks.

## Complete descriptive findings

The 192-cloud run completes in 28.475 seconds, with all original coordinates
and 384 PCA fields exact. It contains 220,513 normal-estimate occurrences over
96 raw frames. Independent scalar verification at 07:49:44 reproduces every
status and representative pixel, with maximum normal-component difference
2.220446049250313e-16. All 174 executing identities and 216 saved source/test
identities match; these are selected source records, not a new hardware test.

Image directions are available at 77.35–88.95% of reviewed locations across
the ten case/capacity strata. Median differences from original ten-neighbor
PCA range 11.99–23.34 degrees. Median two-pixel tangent lengths are roughly
0.52–0.61 nominal mm, while original ten-neighbor support radii have medians
3.49–5.05 mm. These are different geometric scales, not proof either is better.

A separately declared 07:50 follow-up compares source zero to all twenty
remaining stationary indices, both cases and both capacities: eighty frame-pair
occurrences, forty without repeated capacities. It retains every common
representative pixel, unavailable comparison and jointly available subset.
Completed 07:52:44, median direction change on the *same joint subset* is:

| Stationary case / capacity | Image gradient | Original PCA 10 | Original PCA 20 |
| --- | ---: | ---: | ---: |
| A / 900 | 7.543 degrees | 0.792 degrees | 0.642 degrees |
| A / 1800 | 6.597 degrees | 0.862 degrees | 0.650 degrees |
| B / 900 | 6.663 degrees | 0.808 degrees | 0.642 degrees |
| B / 1800 | 6.264 degrees | 1.228 degrees | 0.904 degrees |

The immediate gradient is substantially less repeatable in these nominally
stationary records and loses more available comparisons. This weakens the case
for directly substituting it into tracking; no M27 substitution is declared.
It does not disprove smoothed or larger-support organized normals, which have
not been implemented or tested here. Nominal stationarity does not separate
optical noise, small physical motion and changing neighborhoods, and low
variation can coexist with bias. No new pass threshold or pose is inferred.

Overall, 183,087 image directions are available; 14,623 are missing/outside-band
and 22,803 fail the declared depth-span rule. No reviewed representative is on
the border or numerically degenerate. Stationary comparisons contain 81,587
common-pixel occurrences, 67,054 jointly available for all three methods.
These counts are not independent physical samples.

At 07:54:34, a separate supplement preserves 217 inherited verification files
not in the 216-file selected snapshot. All match their earlier M26 snapshot and
current files, giving 433 distinct preserved source/test/context files. This
post-run supplement is not retroactive process file-access attestation.

The evidence root seals at 07:55:31 UTC: 457 files / 252,676,346 bytes,
manifest `154e6627f123290d30b3307d43ec3593b94cd1244bcf4b0522bc6adf7f147b10`.
The outside-root check under M26 verifies exact inventory and every hash at
07:55:45. No new outputs belong in this sealed root.
