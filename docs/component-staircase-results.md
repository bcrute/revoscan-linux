# Existing staircase: first stopped view — 2026-09-10

Subsequent [more downward view](component-staircase-downward-results.md) is now
complete. Primary spacing moves to about 9.931 mm, and a fixed-pixel control
confirms improved local shape at 10 ms. Patch/direction and physical-reference
uncertainty remain; next is repeated, location-identified calipers, not another
scanner reposition or a new print. The first-view result below is preserved.

## Outcome

Both measured treads return depth in the new view, but the narrow observed
interiors do **not yet give a defensible single calibration height**. The primary
free-plane estimate is about **10.243 mm** against the provisional physical
difference **9.90 mm**. The independently fitted planes differ in angle by about
2.5 degrees; evaluating along the other plane's normal at the other patch's
centroid gives about **10.58 mm**. These are different geometric evaluations
of nonparallel fitted planes, not two interchangeable measurements to average.

The exposure change does not resolve this discrepancy on these patches. The
next useful test is a more downward view exposing broader tread interiors,
using the **same object**, not a new print or a scale correction. Print shape,
caliper contact correspondence and scanner/view-dependent surface distortion
remain unresolved. A successful local plane fit is not an accuracy qualification.

Evidence: `/home/ben/scans/diagnostics/20260910-staircase-check-0aO0EY/`.
See `PROTOCOL.md`, `ANALYSIS-PLAN.md`, `step-results.json`, all 120 frame records,
`step-audit.json`, raw frames, wire trace and the annotated tread-region images.
Sealed **17:16:40 UTC**, **383 files / 72,876,568 bytes** excluding the manifest;
manifest SHA-256 `baefc490af7adfa4a141e476e3f2d771a4c1b84b98030cb0207605a3a71ce649`.
Final document snapshots precede this seal metadata. Closeout verifies all
adapter/acquisition/analysis input bindings and all 449 frozen live/saved source
records unchanged; M27 remains stopped. Do not modify the sealed root.

## Physical reference and acquisition

The [owner's readings](../reference/targets/reconstruction_v1/prints/physical-print-20260831-steps-20260910.json)
are **16.08/25.98 mm from the flat underside to the two highest staircase treads**,
nominally total heights 16/26 mm. They are not heights above the base top or a
measurement of the separate pocket plateau. Subtracting gives a provisional
9.90 mm, assuming a common underside datum. Exact contact locations, repeat
readings and instrument/print uncertainty remain unknown. Do not subtract
nominal base thickness, change the original print record or rescale to this pair.

The owner confirmed the supported/stopped setup. Initial state verified at
**17:04:36 UTC**: exact serial A22704522A6F00A64, frame/exposure 20/20 ms,
manual 0, gain 24, CPU/NPU 58.5/58.2 C. One unchanged guarded-driver bracket
uses exposures **20/10/20 ms**, with frame period 20 ms and gain 24 throughout.
Each stage discards at least 12 frames and one second, then retains 40 frames.

- **160 full deliveries / 120 settled raw frames**, zero sequence drops/shorts.
- Observed whole-run cadence **4.52445 fps**, not a new throughput setting.
- Maximum CPU/NPU **59.8/59.1 C**, including final readback; projector temperature
  is not measured and this short check is not a thermal-equilibrium test.
- Original settings/gain restored and verified after the quiet boundary;
  no transport fault and stable post-close identity. Automatic USB enumeration
  changes 35 to 36 during the documented quiet boundary; no manual replug/reset.
- Raw first/last views retain the same features/pose. First-versus-every-later
  same-stage frame has maximum p90 absolute depth change 0.135/0.090/0.150 mm.
  This supports a stopped-view comparison, not external pose or accuracy truth.

## Fixed surface selection and measurements

The staircase is right of the ramp. Visual depth profiles distinguish the narrow
horizontal treads from the large vertical risers. Two partially overlapping
interiors per tread were fixed after capture but before fitting heights; this
is exploratory surface selection, not a blinded/preregistered accuracy gate.
An initial exploratory column-print command had a SyntaxError before executing;
the corrected tabulation ran without a new capture or measurement-fit retry.

All bounds are x0,y0,x1,y1 exclusive:

| Region | Bounds | Selected pixels |
| --- | --- | ---: |
| Upper primary | 370,136,418,142 | 288 |
| Upper shifted | 378,137,425,143 | 282 |
| Lower primary | 373,183,427,190 | 378 |
| Lower shifted | 383,185,432,191 | 294 |

Every selected pixel returns in all 120 frames at both exposures. No reserved
65535 code occurs in any of the full frames. All four region combinations in
all 120 frames satisfy the unchanged helper's >=200-point, >=2 mm two-direction
support and local planarity checks: **480 measured pairs, zero fit refusals**.
Neither this support nor the return rate establishes metrological correctness.

The independent planes use unchanged factory Pl intrinsics/depth scale. No
calipers, nominal geometry, CAD pose, forced parallelism, temporal averaging,
inter-frame tracking or fitted scale enters a fit. The primary convention is
upper-normal separation evaluated at the retained lower centroid. Reverse is
lower-normal separation evaluated at the retained upper centroid.

| Exposure stage | Primary median (mm) | Reverse median (mm) | Free plane angle median |
| --- | ---: | ---: | ---: |
| Initial 20 ms | 10.243787 | 10.583586 | 2.527 degrees |
| 10 ms | 10.243219 | 10.579105 | 2.500 degrees |
| Final 20 ms | 10.242364 | 10.583875 | 2.534 degrees |

Across the four fixed combinations at 10 ms, upper-normal medians range
**10.227896–10.255512 mm**. Reverse-direction medians range
**10.550007–10.836867 mm**, and median free angles range **2.382–4.130 degrees**.
All alternatives remain recorded; none is chosen for closeness to 9.90 mm.
Within the primary supported lower footprints at 10 ms, fitted separations span
10.04546–10.36764 mm across frames/footprints; this is a geometric variation,
not a calibrated uncertainty interval.

Primary all-selected-point RMS medians, upper/lower in mm, are
0.01992/0.11406 at initial 20 ms, 0.02484/0.11561 at 10 ms, and
0.01957/0.11493 at final 20 ms. Lower exposure does not improve these particular
patches. The lower tread remains much less locally planar than the upper.
The selected strips occupy only six or seven image rows. Edge influence or
surface distortion is plausible, but not isolated from actual print shape by
this result. The approximately +0.343 mm discrepancy is not grounds to apply
a global correction to earlier undersized can/base dimensions.

## Verification and next action

**20 existing preflight/lifecycle/geometry tests** pass before hardware;
**7 reporting/selection tests** pass afterward. Numerical analysis takes
0.96947 seconds. The audit checks all 133 bound inputs, all 120 unique raw
frames, 480 unique patch fits, 480 pair results, **149,040 ordered selected
pixel samples including overlap**, all-point RMS and stage scalar summaries.
It recomputes the unchanged helper; it is not independent calibration truth.

Next operator action: raise/reposition the scanner to look more downward at
the staircase while keeping it supported and the turntable stopped. Aim for
both measured tread tops to occupy broad visible areas rather than thin strips;
roughly 35–45 degrees downward is a setup guide, not a measured pose requirement.
The underside, pocket floor and complete object outline need not be visible
for this local check. Keep roughly the current distance if practical.

After the next setup, check framing/support before judging the provisional
spacing. Repeated location-matched calipers and reference uncertainty will still
be needed for an accuracy decision. No acceptance gate, production recipe,
calibration, firmware, mesh or suspended M27 change is promoted here.
