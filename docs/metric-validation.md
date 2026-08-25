# Metric validation: the depth scale is millimetres

Status: measured 2026-08-24 against a caliper-measured target, with a tolerance
declared before each comparison. This closes the substantive part of M4. Every
frame is committed under `tests/fixtures/metric/` with a manifest carrying the
ground truth, the surface condition and each frame's sha256.

## The headline

**The depth scale is millimetres.** The float32 that decodes to `1/200` has been
in this repository since scanner-008 with its physical meaning unknown, and
every record the code writes has said so. A 58.0 mm object measures 57.7 mm.
That rules out centimetres, arbitrary units, or anything else by orders of
magnitude.

**The best clean measurement is −0.49%**, from a sprayed target at 241.5 mm with
a residual rms of 0.098.

## The target

An aluminium drink can, standing upright. The body measures a consistent
**58.0 mm** on calipers, five readings, uncertainty **0.05 mm**. The flanged top
and bottom stand proud of the body and were cropped where they entered frame.

A cylinder was used rather than the textbook sphere because the owner had one
and a sphere works only if you own a sphere. Orientation is not optional: an
upright cylinder reveals a principal-point shear by 161% of its radius, one
lying across the view by 0.0025 — no signal at all. See
[capture-checklist.md](capture-checklist.md).

## Every measurement

| Surface | Standoff | Diameter | Error | Yield | Residual rms | Usable |
| --- | --- | --- | --- | --- | --- | --- |
| bare | 98.1 | 56.689 | −2.26% | 26.2% | 0.16 | yes |
| bare | 122.7 | 57.705 | −0.51% | 40.0% | 0.21 | yes |
| bare | 244.1 | 57.445 | −0.96% | 23.1% | 0.22 | yes |
| sprayed | ~95 | — | — | **0.8%** | — | **saturated** |
| sprayed | ~125 | 62.01 | +6.92% | 17.4% | **1.41** | **rejected** |
| **sprayed** | **241.5** | **57.716** | **−0.49%** | 28.1% | **0.098** | **best** |

Standoffs are what the scanner itself measured, not what was estimated with a
ruler. The owner's estimates were never used in any calculation.

## Four things this established that were not known this morning

### The unit is millimetres

See above. This was the single largest unknown in the project.

### A bare specular surface outperforms a sprayed one at close range

This is the opposite of the standard advice, and the mechanism is exposure.
`id_msg_cam_get_auto_expo_mode` reads **0** on this unit — automatic exposure is
**off** — and `id_msg_cam_get_exposure_time` is pinned at 20000. A matte white
coating close in returns far more light than bare metal, the sensor saturates,
and the fringe decode fails:

| Sprayed target | Yield |
| --- | --- |
| 95 mm | 0.8% |
| ~125 mm | 17.4% |
| 241.5 mm | 28.1% |

Monotonic with distance, exactly as illumination falling off with the square
predicts. Bare aluminium gave **40%** at 122.7 mm — the highest yield of any
frame in this repository — because the specular lobe was throwing most of the
light away from the camera and keeping the sensor in range.

Until exposure control exists, **spray helps at range and hurts close in.**

### The structured residual on bare metal was largely optical, not geometric

At the same distance and pose, changing only the surface:

| | shear correlation |
| --- | --- |
| bare, 244.1 | **−0.21** |
| sprayed, 241.5 | **+0.13** |

A geometric shear from a wrong principal point cannot change sign when you paint
the target — `cx` belongs to the lens, not the object. So the residual measured
on bare aluminium was substantially a specular artifact.

The mechanism is sampling bias rather than measurement error: on a curved
specular surface the lobe throws light away from the camera over part of the
arc, so the surviving returns favour particular incidence angles. The shear
indicator names no cause by design, and this is a clean demonstration of why
that caution is warranted.

Two earlier readings are weakened by this and are withdrawn: that the detected
shear said anything specific about `cx`, and that the bare-metal error curve's
minimum near 120 mm was a property of the sensor's calibration. The sprayed data
cannot confirm that minimum, because the sprayed capture at ~125 mm saturated.

### The harness refused the bad measurement rather than reporting it

The sprayed capture at ~125 mm fitted a diameter of 62.01 — 6.92% out. It was
never reported as a result. `fit_uncertainty` came to 2.82 against a declared
tolerance of 0.5, and the verdict was INCONCLUSIVE with the reason *"a pass was
unreachable before the target was even compared."*

That is the design working. A harness that had reported +6.92% would have been
worse than no harness.

## What is still not established

- **A calibration correction.** Errors range from −0.49% to −2.26% across
  distance and surface, and the two candidate causes tested — a constant scale
  error and a constant principal-point shear — are both ruled out by their
  distance signatures. No model is offered.
- **Whether the bare curve's minimum near 120 mm is real.** The sprayed
  comparison at that distance saturated.
- **Anything about other objects, orientations, or positions in the frame.** One
  can, one tilt, one place in the image.
- **Lens distortion.** The deprojection remains an ideal pinhole with no
  distortion term. The far captures show a smooth symmetric axial bow of about
  0.19 peak-to-peak that is consistent with radial distortion about the
  principal point, but that is a hint and not a measurement.

## Two hardware faults found on the way

Both are in [capture-checklist.md](capture-checklist.md) with numbers.
**Leave 5 seconds between captures** — back-to-back gives 1 of 3 complete, and
every failure truncates at exactly the same row. **Do not use a USB hub** — 28
link drops on one, zero after moving to a root port. Neither biased the frames
that completed: the same can measured 56.641 through the failing hub and 56.689
direct, inside the frame-to-frame spread.
