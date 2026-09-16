# Cue-ball scale check — 2026-09-11

A sphere reference, fitted in single frames with `tools/sphere_reference.py`
(tests: `tests/test_sphere_reference.py`). Reference object: the owner's cue
ball, calipers 57.25–57.28 mm at several orientations, taken as 57.265 mm.
Nothing here changes production code, calibration files or the device.

## What was measured

| Capture | Evidence root | Result |
| --- | --- | --- |
| Stationary ball, 20/10/7/20 ms exposure bracket, 40 settled frames each | `/home/ben/scans/diagnostics/20260911-cueball-bracket-2930410420/` | Sphere diameter 56.98 mm at every exposure (−0.50%), frame sigma 0.003 mm; fringe ripple on the ball 0.002–0.004 mm rms; residual pattern identical at all exposures |
| Same ball turned a quarter turn (moved ~12 mm), 20/10/20 ms | `/home/ben/scans/diagnostics/20260911-cueball-rotated-138616680/` | 57.04 mm (−0.40%); the horizontal-over-vertical residual pattern stayed in the image frame (long axis +15° then −6° from image horizontal), so it belongs to the scanner |
| Cue ball plus striped ball on the slow turntable, 150 and 256 frames, default settings, no control writes | `/home/ben/scans/diagnostics/20260911-ballpair-turn-2006232181/`, `/home/ben/scans/diagnostics/20260911-ballpair-longturn-69468320/` | Cue-ball orbit over 163° is a circle to 0.02 mm rms, axis ratio 1.0002: camera x and the table's depth direction share one scale to better than 0.05% |

An earlier attempt through `tools/scanner_timing_probe.py` hung the device's
USB stack at the stage transition (control writes right after a stream close)
and needed a physical replug; the brackets above used the single-stream
`quiet_capture.py` helper from the pocket-exposure evidence root instead.

## Interpretation

Rendering a true 57.265 mm sphere at the measured pose through the same
pipeline (`scanner3d.simulate.render`, factory intrinsics) with per-axis
measured-over-true factors reproduces both the 56.98 mm sphere fit and the
1.004 horizontal-over-vertical ellipsoid ratio only for approximately:

| Axis | Measured over true |
| --- | ---: |
| x (image horizontal) | 0.9985 |
| y (image vertical) | 0.9945 |
| z (depth) | 0.9985 |

A sphere fit follows the apex curvature, so a vertical-only shortfall of 0.55%
shows up amplified in the sphere diameter; the isotropic part is only about
0.15%. Exposure is irrelevant for this object (the glossy ball never
saturates); the undersize is geometric.

Applying the implied correction — `fy` × 0.996 (895.08 px), depth scale
× 1.0015 (0.0050075 mm/code), `fx` unchanged — to the real frames gives:

| Data | Corrected sphere diameter | vs 57.265 |
| --- | ---: | ---: |
| First pose, z ≈ 209 mm | 57.280 mm, x/y 1.0001 | +0.03% |
| Rotated pose, z ≈ 221 mm | 57.330 mm | +0.11% |
| Turn, cue ball, 133 frames | 57.362 mm | +0.17% |

The residual +0.1 to +0.2% grows with distance, so a small range-dependent term
remains; it is inside a 0.15 mm + 0.2% tolerance on this object. Mapping it
would take the same ball at three or four distances.

## What this does and does not settle

- The raw depth, single frame, is within about 0.2% after a two-number
  correction. The 0.8% undersize of the fused printed platform is therefore
  not the sensor's; look in registration, fusion and the width measurement.
- The two-ball spacing was the weakest measurement: the striped ball's white
  polar cap is partial and its band edge contaminates the fit (56.64 mm
  against its 57.00 mm calipers), and the balls sat about 0.3 mm apart on the
  cloth. Its per-direction factors are recorded in the session reports but
  nothing above rests on them.
- The vertical factor rests on the image-fixed residual pattern and the
  render model; a stacked pair or a scanner rolled 90° would measure it
  directly.
- The turntable ran at about 550 frames per turn during these captures.

## Applied

Later the same day the correction was moved into the product as
`scanner3d.metric_correction.CUE_BALL_20260911`: measured-over-true x 0.9985,
y 0.9945, z 0.9985, applied as `fy` × 0.9960, depth scale × 1.0015, `fx`
unchanged. The window, `scanner3d reconstruct` and `tools/target_gate.py`
deproject every frame under it; `--no-metric-correction` on the latter two is
the control, and each report names the deprojection it used. The opening
paragraph stands for the measurement itself: nothing in it changed the device or
the factory record, and nothing does now - the correction is applied to the
frames as they are read. See [metric-validation.md](metric-validation.md).
